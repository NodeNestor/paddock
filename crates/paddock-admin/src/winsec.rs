//! Windows pipe security: a DACL granting access to the **owning user + SYSTEM
//! only** (doc §5.1). Windows named pipes default to a DACL that lets Everyone
//! read - and, combined with pipes accepting NETWORK clients by default, that
//! is exactly the exposed-/slots class of leak we refuse to ship. Both are
//! closed here: this SDDL descriptor for the DACL, and
//! `reject_remote_clients(true)` (= `PIPE_REJECT_REMOTE_CLIENTS`) at creation.
//!
//! Flow: current process token -> user SID -> SDDL string
//! `D:P(A;;GA;;;SY)(A;;GA;;;<sid>)` (protected DACL, GENERIC_ALL for SYSTEM
//! and the user, nothing else) -> SECURITY_ATTRIBUTES for pipe creation.

use std::ffi::c_void;
use std::io;

use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, LocalFree};
use windows_sys::Win32::Security::Authorization::{
    ConvertSidToStringSidW, ConvertStringSecurityDescriptorToSecurityDescriptorW, SDDL_REVISION_1,
};
use windows_sys::Win32::Security::{
    GetTokenInformation, SECURITY_ATTRIBUTES, TOKEN_QUERY, TOKEN_USER, TokenUser,
};
use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

/// Owns the security descriptor memory (LocalAlloc'd by the SDDL conversion)
/// and hands out `SECURITY_ATTRIBUTES` pointers for pipe creation. Must
/// outlive every create call that uses it.
pub struct PipeSecurity {
    descriptor: *mut c_void,
    attrs: SECURITY_ATTRIBUTES,
}

// The descriptor is immutable after construction and only read by pipe
// creation calls; moving the owner across threads is fine.
unsafe impl Send for PipeSecurity {}

impl PipeSecurity {
    /// Build the user+SYSTEM-only security attributes for this process's user.
    pub fn user_only() -> io::Result<Self> {
        let sid = current_user_sid_string()?;
        // Protected (P) DACL: no inherited ACEs can widen it. GA = GENERIC_ALL
        // for SYSTEM (SY) and the owning user; no other ACEs -> no other access.
        let sddl = format!("D:P(A;;GA;;;SY)(A;;GA;;;{sid})");
        let mut wide: Vec<u16> = sddl.encode_utf16().chain(std::iter::once(0)).collect();
        let mut descriptor: *mut c_void = std::ptr::null_mut();
        let ok = unsafe {
            ConvertStringSecurityDescriptorToSecurityDescriptorW(
                wide.as_mut_ptr(),
                SDDL_REVISION_1,
                &mut descriptor as *mut *mut c_void as *mut _,
                std::ptr::null_mut(),
            )
        };
        if ok == 0 {
            return Err(io::Error::last_os_error());
        }
        let attrs = SECURITY_ATTRIBUTES {
            nLength: std::mem::size_of::<SECURITY_ATTRIBUTES>() as u32,
            lpSecurityDescriptor: descriptor,
            bInheritHandle: 0,
        };
        Ok(PipeSecurity { descriptor, attrs })
    }

    /// Raw pointer for `create_with_security_attributes_raw`. Valid as long as
    /// `self` lives.
    pub fn as_ptr(&self) -> *mut c_void {
        &self.attrs as *const SECURITY_ATTRIBUTES as *mut c_void
    }
}

impl Drop for PipeSecurity {
    fn drop(&mut self) {
        unsafe {
            LocalFree(self.descriptor);
        }
    }
}

/// The current process user's SID in string form (e.g. `S-1-5-21-...`).
fn current_user_sid_string() -> io::Result<String> {
    unsafe {
        let mut token: HANDLE = std::ptr::null_mut();
        if OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) == 0 {
            return Err(io::Error::last_os_error());
        }
        // First call sizes the TOKEN_USER buffer; second fills it.
        let mut len: u32 = 0;
        GetTokenInformation(token, TokenUser, std::ptr::null_mut(), 0, &mut len);
        let mut buf = vec![0u8; len as usize];
        let ok = GetTokenInformation(
            token,
            TokenUser,
            buf.as_mut_ptr() as *mut c_void,
            len,
            &mut len,
        );
        CloseHandle(token);
        if ok == 0 {
            return Err(io::Error::last_os_error());
        }
        let user = &*(buf.as_ptr() as *const TOKEN_USER);
        let mut sid_w: *mut u16 = std::ptr::null_mut();
        if ConvertSidToStringSidW(user.User.Sid, &mut sid_w) == 0 {
            return Err(io::Error::last_os_error());
        }
        // Copy out of the LocalAlloc'd wide string, then free it.
        let mut n = 0usize;
        while *sid_w.add(n) != 0 {
            n += 1;
        }
        let s = String::from_utf16_lossy(std::slice::from_raw_parts(sid_w, n));
        LocalFree(sid_w as *mut c_void);
        Ok(s)
    }
}
