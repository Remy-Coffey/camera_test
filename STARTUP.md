# Backend Startup Notes

## Standard startup

Use this for normal testing. It does not enable hot reload, so the process is cleaner and easier to stop.

```powershell
cd E:\camera_test
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\run_backend.ps1
```

## Development startup with hot reload

Only use this when you are actively editing backend code and want automatic reload.

```powershell
cd E:\camera_test
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\run_backend.ps1 -Reload
```

## Difference

- Default mode: no `--reload`, fewer child processes, easier to close.
- `-Reload`: development mode, enables `watchfiles`, may spawn extra processes and can be harder to stop cleanly.

## If the process does not close cleanly

Open a new PowerShell window and run:

```powershell
Get-Process py,python -ErrorAction SilentlyContinue | Stop-Process -Force
```
