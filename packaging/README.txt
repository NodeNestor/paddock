Paddock - Windows x64
=====================

A low-latency, high-throughput serving engine for open-source language models,
running on your own NVIDIA hardware.

Start here
----------

  1. Unzip this folder anywhere you like - it is fully portable.
  2. Run paddock.exe (add --host 0.0.0.0 if you want to bind to your non localhost)
  3. Open https://localhost:11500

That is the Studio: download models, start them, chat with them, and watch what
the GPU is doing.

The first visit shows a certificate warning. That is expected - paddock serves
the Studio over HTTPS using a certificate it generated on THIS machine, which no
public authority has signed (nor should one: it is for a name that means "this
box"). The Studio's trust page will install the root if you would rather not see
the warning again.


What the two programs are
-------------------------

  paddock.exe          the manager and the Studio. This is the one you run.
                       It downloads models, starts and stops model servers, and
                       serves the web UI. It does NO inference itself.

  paddock-runner.exe   serves ONE model on ONE port, headless. The manager
                       starts these for you; you never have to. It is a normal
                       program, so you can also run it by hand with a config
                       file and no manager at all.

Clients talk to the runners DIRECTLY - the manager is never a proxy, so nothing
sits between your requests and the GPU.


What you need
-------------

  Windows 10/11 x64, an NVIDIA GPU, and NVIDIA driver 580 or newer.

580 is a hard floor, not a recommendation: paddock is built against the CUDA
13.0 driver API, and an older driver refuses to load it. Any driver from the
580 branch onwards works - there is no CUDA toolkit to install, only the
display driver.

Nothing else. No Python, no CUDA toolkit, no Visual C++ redistributable - the
kernels are compiled into paddock-runner.exe, and both binaries link the C
runtime statically, so there is no redistributable to install.

Supported GPUs - and "supported" here means measured on that die, not merely
compiled for:

  Blackwell   RTX PRO 6000 / 5000 / 4500 / 4000 / 2000,
              GeForce RTX 5090 through 5050, B200 / GB200
  Ampere      RTX A6000 / A5000 / A4000 / A2000, A40 / A10,
              GeForce RTX 3090 / 3080

Ada Lovelace (RTX 40-series, RTX 6000 Ada, L40S) is not validated yet. The
kernels for it ARE in this build and it will very likely run, but its bring-up
is unfinished, so paddock refuses rather than serve numbers nobody has checked.
Set PADDOCK_UNVALIDATED_ARCH=1 to try it anyway; those runs are stamped
UNVALIDATED in the logs.

Hopper (H100, H200, GH200) is not in this build at all - no kernels ship for
it, so the override above will not help. Anything older than Ampere will not
work at all.


Using it from your own code
---------------------------

Each model server speaks the OpenAI and Anthropic APIs.
Point an existing client at the runner's port and change nothing else:

  OpenAI       base_url = "http://localhost:<port>/v1"
  Anthropic    base_url = "http://localhost:<port>"

The Studio shows each server's port and its API key. Servers bound to localhost
can be opened without a key; anything reachable from the network always needs
one.


Where your things live
----------------------

Everything paddock manager creates is in the data\ folder beside these binaries -
models, database, settings, logs, certificates. Move the whole folder and the
install moves with it; delete it and paddock starts fresh.

Models are stored as plain GGUF files, in ordinary folders, under whatever name
they were published with. Nothing is renamed into a hashed blob store, so an
existing model collection works as-is: point paddock at the folder in Studio
settings. See data\README.txt for the full layout.


Licence
-------

Paddock is free and open-source software, dual-licensed under the MIT licence
and the Apache Licence 2.0 - take whichever of the two suits you, you do not
have to comply with both. LICENSE-MIT.txt and LICENSE-APACHE.txt contain the
full texts and control over this summary.

Both licences let you use, copy, modify and redistribute paddock, for any
purpose, commercial included, provided you keep the copyright and licence
notices with it. Apache-2.0 additionally grants an explicit patent licence,
which some adopters require.

THIRD-PARTY-NOTICES.txt lists every third-party component in these binaries and
its licence - including PDFium, which is built from source and linked in rather
than shipped as a sidecar. Those components remain under their own licences.


  https://truespar.com/paddock
