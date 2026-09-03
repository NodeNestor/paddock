//! Narrow a path to the user running paddock.
//!
//! `ca.key` can mint a certificate for any name the CA's constraints allow, so
//! it is the one genuinely sensitive file the manager writes. A data root can
//! land anywhere - portable mode puts it beside the exe, which may be a shared
//! drive with permissive inherited ACLs - so inheriting whatever the parent
//! directory happens to grant is not good enough.
//!
//! Best effort by design: every failure is silent and the caller carries on.
//! A box that cannot tighten a file's ACL should still come up with https
//! working, because the alternative is falling back to cleartext - strictly
//! worse for the same user.

use std::path::Path;

#[cfg(unix)]
pub fn restrict_to_owner(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let Ok(meta) = std::fs::metadata(path) else {
        return;
    };
    // 0700 for a directory (x = the right to traverse it), 0600 for a file.
    let mode = if meta.is_dir() { 0o700 } else { 0o600 };
    let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode));
}

/// Replace the object's DACL with a **protected** one naming only SYSTEM and
/// the owning user - the same descriptor the runner's admin pipe uses
/// (`paddock_admin::winsec`), for the same reason. Protected matters as much
/// as the ACEs: without it, inherited ACEs from the parent directory are
/// merged back in and a permissive parent silently re-widens the file.
#[cfg(windows)]
pub fn restrict_to_owner(path: &Path) {
    use std::ffi::c_void;
    use std::os::windows::ffi::OsStrExt;

    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, LocalFree};
    use windows_sys::Win32::Security::Authorization::{
        ConvertSidToStringSidW, ConvertStringSecurityDescriptorToSecurityDescriptorW,
        SDDL_REVISION_1, SE_FILE_OBJECT, SetNamedSecurityInfoW,
    };
    use windows_sys::Win32::Security::{
        ACL, DACL_SECURITY_INFORMATION, GetSecurityDescriptorDacl,
        PROTECTED_DACL_SECURITY_INFORMATION,
    };

    let Some(sid) = current_user_sid() else {
        return;
    };
    let sddl = format!("D:P(A;OICI;GA;;;SY)(A;OICI;GA;;;{sid})");
    let mut sddl_w: Vec<u16> = sddl.encode_utf16().chain(std::iter::once(0)).collect();
    let mut path_w: Vec<u16> = path
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();

    unsafe {
        let mut sd: *mut c_void = std::ptr::null_mut();
        if ConvertStringSecurityDescriptorToSecurityDescriptorW(
            sddl_w.as_mut_ptr(),
            SDDL_REVISION_1,
            &mut sd as *mut *mut c_void as *mut _,
            std::ptr::null_mut(),
        ) == 0
        {
            return;
        }
        // SetNamedSecurityInfoW wants the ACL out of the descriptor, not the
        // descriptor itself.
        let mut dacl: *mut ACL = std::ptr::null_mut();
        let mut present = 0;
        let mut defaulted = 0;
        if GetSecurityDescriptorDacl(sd, &mut present, &mut dacl, &mut defaulted) != 0
            && present != 0
        {
            SetNamedSecurityInfoW(
                path_w.as_mut_ptr(),
                SE_FILE_OBJECT,
                DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                dacl,
                std::ptr::null_mut(),
            );
        }
        LocalFree(sd);
    }

    /// The current process user's SID in string form.
    fn current_user_sid() -> Option<String> {
        use windows_sys::Win32::Security::{
            GetTokenInformation, TOKEN_QUERY, TOKEN_USER, TokenUser,
        };
        use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

        unsafe {
            let mut token: HANDLE = std::ptr::null_mut();
            if OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) == 0 {
                return None;
            }
            let mut len: u32 = 0;
            GetTokenInformation(token, TokenUser, std::ptr::null_mut(), 0, &mut len);
            let mut buf = vec![0u8; len as usize];
            let ok = GetTokenInformation(token, TokenUser, buf.as_mut_ptr().cast(), len, &mut len);
            CloseHandle(token);
            if ok == 0 {
                return None;
            }
            let user = &*(buf.as_ptr() as *const TOKEN_USER);
            let mut sid_w: *mut u16 = std::ptr::null_mut();
            if ConvertSidToStringSidW(user.User.Sid, &mut sid_w) == 0 {
                return None;
            }
            let mut n = 0usize;
            while *sid_w.add(n) != 0 {
                n += 1;
            }
            let s = String::from_utf16_lossy(std::slice::from_raw_parts(sid_w, n));
            LocalFree(sid_w.cast());
            Some(s)
        }
    }
}

#[cfg(not(any(unix, windows)))]
pub fn restrict_to_owner(_path: &Path) {}
