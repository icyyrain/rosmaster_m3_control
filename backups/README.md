# Local backup policy

`backups/remote/` contains snapshots copied from the Jetson before firmware or
autostart changes. The payload is intentionally ignored by Git because it may
be large and may contain third-party files. Add a dated manifest and checksum
file beside each snapshot when a backup is created.

An STM32 rollback image and the matching factory source project are still
required before the control board is flashed.

