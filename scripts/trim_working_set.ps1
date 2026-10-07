# Trim the largest python process' working set into the pagefile.
# Keeps free *physical* RAM above the harness watchdog threshold on a
# RAM-constrained machine. Math-neutral: paged pages fault back on access
# (pagefile has ~20GB headroom); pinned/CUDA memory is never paged.
$sig = '[DllImport("psapi.dll")] public static extern bool EmptyWorkingSet(IntPtr hProcess);'
$p = Add-Type -MemberDefinition $sig -Name PsApi -Namespace W32 -PassThru
$proc = Get-Process python -ErrorAction SilentlyContinue |
        Sort-Object WS -Descending | Select-Object -First 1
if ($proc -and $proc.WS -gt 500MB) {
    $mb = [int]($proc.WS / 1MB)
    $p::EmptyWorkingSet($proc.Handle) | Out-Null
    "trimmed pid $($proc.Id) from ${mb}MB at $(Get-Date -Format HH:mm:ss)"
} else {
    "nothing to trim at $(Get-Date -Format HH:mm:ss)"
}
