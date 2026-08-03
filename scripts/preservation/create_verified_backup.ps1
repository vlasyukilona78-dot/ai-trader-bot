[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("LocalCheckpoint", "DisasterResilient")]
    [string]$Mode,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$BackupBase,

    [string]$RootPath,
    [string]$MexcPath,
    [string]$PythonPath,
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Stop-Preservation {
    param([Parameter(Mandatory = $true)][string]$Message)
    throw "STOP: $Message"
}

function Get-NormalizedRelativePath {
    param([Parameter(Mandatory = $true)][string]$Path)
    return ($Path -replace "\\", "/").TrimStart("/")
}

function Get-RelativePathUnderRoot {
    param(
        [Parameter(Mandatory = $true)][string]$BasePath,
        [Parameter(Mandatory = $true)][string]$FullPath
    )

    $base = [IO.Path]::GetFullPath($BasePath).TrimEnd("\")
    $candidate = [IO.Path]::GetFullPath($FullPath)
    $prefix = $base + "\"
    if (-not $candidate.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        Stop-Preservation "path escapes source root: $candidate"
    }
    return Get-NormalizedRelativePath $candidate.Substring($prefix.Length)
}

function Test-PathInside {
    param(
        [Parameter(Mandatory = $true)][string]$Candidate,
        [Parameter(Mandatory = $true)][string]$Container
    )

    $candidateFull = [IO.Path]::GetFullPath($Candidate).TrimEnd("\")
    $containerFull = [IO.Path]::GetFullPath($Container).TrimEnd("\")
    if ($candidateFull.Equals($containerFull, [StringComparison]::OrdinalIgnoreCase)) {
        return $true
    }
    return $candidateFull.StartsWith(
        $containerFull + "\",
        [StringComparison]::OrdinalIgnoreCase
    )
}

function Get-NearestExistingDirectory {
    param([Parameter(Mandatory = $true)][string]$Path)

    $cursor = [IO.Path]::GetFullPath($Path)
    while (-not (Test-Path -LiteralPath $cursor)) {
        $parent = Split-Path -Parent $cursor
        if ([string]::IsNullOrWhiteSpace($parent) -or $parent -eq $cursor) {
            Stop-Preservation "no existing ancestor for destination: $Path"
        }
        $cursor = $parent
    }

    $item = Get-Item -LiteralPath $cursor -Force
    if (-not $item.PSIsContainer) {
        $cursor = Split-Path -Parent $cursor
    }
    return [IO.Path]::GetFullPath($cursor)
}

function Assert-DestinationAncestorsSafe {
    param([Parameter(Mandatory = $true)][string]$Path)

    $cursor = Get-NearestExistingDirectory $Path
    while (-not [string]::IsNullOrWhiteSpace($cursor)) {
        $item = Get-Item -LiteralPath $cursor -Force
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            Stop-Preservation "destination ancestor is a reparse point: $cursor"
        }
        if (Test-Path -LiteralPath (Join-Path $cursor ".git")) {
            Stop-Preservation "destination is inside a Git repository: $cursor"
        }
        if ((Test-Path -LiteralPath (Join-Path $cursor "HEAD") -PathType Leaf) -and
            (Test-Path -LiteralPath (Join-Path $cursor "objects") -PathType Container) -and
            (Test-Path -LiteralPath (Join-Path $cursor "refs") -PathType Container)) {
            Stop-Preservation "destination is inside a bare Git repository: $cursor"
        }

        $parent = Split-Path -Parent $cursor
        if ([string]::IsNullOrWhiteSpace($parent) -or $parent -eq $cursor) {
            break
        }
        $cursor = $parent
    }
}

function Invoke-GitLines {
    param(
        [Parameter(Mandatory = $true)][string]$Repository,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = @(& git -c core.quotepath=false -C $Repository @Arguments 2>$null)
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($exitCode -ne 0) {
        Stop-Preservation "Git command failed in $Repository"
    }
    return @($output | ForEach-Object { [string]$_ })
}

function Test-GitRepository {
    param([Parameter(Mandatory = $true)][string]$Repository)
    $result = @(Invoke-GitLines $Repository @("rev-parse", "--is-inside-work-tree"))
    return $result.Count -eq 1 -and $result[0].Trim() -eq "true"
}

function Get-ExclusionReason {
    param([Parameter(Mandatory = $true)][string]$RelativePath)

    $path = Get-NormalizedRelativePath $RelativePath
    $parts = @($path -split "/")
    $leaf = $parts[$parts.Count - 1]

    if ($leaf -match "(?i)^\.env(?:\..*)?$") { return "environment-secret" }
    if ($path -match "(?i)(^|/)logs/system\.log$") { return "runtime-system-log" }
    if ($path -match "(?i)(^|/)data/runtime/bot_runtime\.lock$") { return "stale-runtime-lock" }
    if ($leaf -match "(?i)\.db-(?:wal|shm)$") { return "raw-sqlite-sidecar" }
    if ($leaf -match "(?i)\.(?:pem|key|p12|pfx)$" -or
        $leaf -match "(?i)^(?:id_rsa|id_ed25519)$") {
        return "private-key-material"
    }
    if ($leaf -match "(?i)^recovery_snapshot_.*\.zip$") { return "recovery-artifact" }

    foreach ($part in $parts) {
        if ($part -eq ".git" -or $part -eq ".idea") { return "repository-local-metadata" }
        if ($part -match "(?i)^\.venv(?:_|$)" -or $part -eq "venv") {
            return "virtual-environment"
        }
        if ($part -match "(?i)^(?:__pycache__|\.pytest_cache|\.mypy_cache|\.ruff_cache|cache|caches)$") {
            return "cache"
        }
        if ($part -match "(?i)^(?:recovery|recovery_|\.git_corrupt_|\.git_mixed_).*") {
            return "recovery-artifact"
        }
    }

    if ($leaf -match "(?i)\.db$") { return "sqlite-online-backup" }
    return $null
}

function Test-TextCandidate {
    param([Parameter(Mandatory = $true)][string]$RelativePath)

    $path = Get-NormalizedRelativePath $RelativePath
    $leaf = [IO.Path]::GetFileName($path)
    $extension = [IO.Path]::GetExtension($leaf).ToLowerInvariant()
    if ($leaf -in @(".gitignore", ".gitattributes", "requirements.txt", "CLAUDE.md", "AGENTS.md")) {
        return $true
    }
    if ($leaf -match "(?i)\.env\.example$") { return $true }
    return $extension -in @(
        ".py", ".ps1", ".psm1", ".psd1", ".json", ".jsonl", ".yaml", ".yml",
        ".toml", ".ini", ".cfg", ".conf", ".md", ".txt", ".csv", ".lock",
        ".xml", ".html", ".js", ".ts"
    )
}

$script:SecretRules = @(
    [pscustomobject]@{ Name = "private-key-header"; Pattern = "-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----" },
    [pscustomobject]@{ Name = "aws-access-key"; Pattern = "\b(?:AKIA|ASIA)[0-9A-Z]{16}\b" },
    [pscustomobject]@{ Name = "github-token"; Pattern = "\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})\b" },
    [pscustomobject]@{ Name = "openai-token"; Pattern = "\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}\b" },
    [pscustomobject]@{ Name = "telegram-token"; Pattern = "\b[0-9]{8,12}:(?!ABCDEFGHIJKLMNOPQRSTUVWXYZ|abcdefghijklmnopqrstuvwxyz|0123456789)[A-Za-z0-9_-]{30,}\b" },
    [pscustomobject]@{ Name = "slack-token"; Pattern = "\bxox[baprs]-[A-Za-z0-9-]{20,}\b" },
    [pscustomobject]@{
        Name = "literal-credential-assignment"
        Pattern = '(?i)\b(?:api[_-]?key|api[_-]?secret|bot[_-]?token|access[_-]?token|password)\b\s*[:=]\s*[''"](?!your_|replace_|example|changeme|<|\$\{)[^''"\s]{16,}[''"]'
    }
)

function Assert-LinesContainNoSecrets {
    param(
        [Parameter(Mandatory = $true)][object[]]$Lines,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $lineNumber = 0
    foreach ($lineValue in $Lines) {
        $lineNumber += 1
        $line = [string]$lineValue
        foreach ($rule in $script:SecretRules) {
            if ($line -match $rule.Pattern) {
                Stop-Preservation "secret scan hit rule=$($rule.Name) source=$Label line=$lineNumber"
            }
        }
    }
}

function Assert-FilesContainNoSecrets {
    param(
        [Parameter(Mandatory = $true)][string]$BasePath,
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][string[]]$RelativePaths,
        [Parameter(Mandatory = $true)][string]$Label
    )

    foreach ($relative in $RelativePaths) {
        if (-not (Test-TextCandidate $relative)) { continue }
        $fullPath = Join-Path $BasePath ($relative -replace "/", "\")
        if (-not (Test-Path -LiteralPath $fullPath -PathType Leaf)) { continue }
        $lineNumber = 0
        Get-Content -LiteralPath $fullPath -Encoding UTF8 -ReadCount 1 | ForEach-Object {
            $lineNumber += 1
            $line = [string]$_
            foreach ($rule in $script:SecretRules) {
                if ($line -match $rule.Pattern) {
                    Stop-Preservation "secret scan hit rule=$($rule.Name) source=$Label/$relative line=$lineNumber"
                }
            }
        }
    }
}

function Get-GitInventory {
    param(
        [Parameter(Mandatory = $true)][string]$Repository,
        [Parameter(Mandatory = $true)][string]$Label
    )

    if (-not (Test-GitRepository $Repository)) {
        Stop-Preservation "$Label is not a Git worktree: $Repository"
    }
    if (@(Invoke-GitLines $Repository @("ls-files", "-u")).Count -ne 0) {
        Stop-Preservation "$Label has unmerged index entries"
    }

    $tracked = @(Invoke-GitLines $Repository @("ls-files") | Where-Object { $_ -ne "" })
    $untracked = @(Invoke-GitLines $Repository @("ls-files", "--others", "--exclude-standard") | Where-Object { $_ -ne "" })
    $dirty = @(Invoke-GitLines $Repository @("diff", "--name-only", "HEAD", "--") | Where-Object { $_ -ne "" })
    $staged = @(Invoke-GitLines $Repository @("diff", "--cached", "--name-only", "HEAD", "--") | Where-Object { $_ -ne "" })

    $trackedSafe = New-Object System.Collections.Generic.List[string]
    $untrackedSafe = New-Object System.Collections.Generic.List[string]
    $dirtySafe = New-Object System.Collections.Generic.List[string]
    $sqlite = New-Object System.Collections.Generic.List[string]
    $deleted = New-Object System.Collections.Generic.List[string]
    $excluded = New-Object System.Collections.Generic.List[object]

    foreach ($pathValue in $tracked) {
        $path = Get-NormalizedRelativePath $pathValue
        $reason = Get-ExclusionReason $path
        if ($reason -eq "private-key-material") {
            Stop-Preservation "$Label contains private-key-like tracked path: $path"
        }
        if ($reason -eq "sqlite-online-backup") {
            if (Test-Path -LiteralPath (Join-Path $Repository ($path -replace "/", "\")) -PathType Leaf) {
                [void]$sqlite.Add($path)
            }
            continue
        }
        if ($null -ne $reason) {
            [void]$excluded.Add([pscustomobject]@{ Path = $path; Reason = $reason; Source = "tracked" })
            continue
        }

        $fullPath = Join-Path $Repository ($path -replace "/", "\")
        if (-not (Test-Path -LiteralPath $fullPath -PathType Leaf)) {
            [void]$deleted.Add($path)
            continue
        }
        [void]$trackedSafe.Add($path)
    }

    foreach ($pathValue in $untracked) {
        $path = Get-NormalizedRelativePath $pathValue
        $reason = Get-ExclusionReason $path
        if ($reason -eq "private-key-material") {
            Stop-Preservation "$Label contains private-key-like untracked path: $path"
        }
        if ($reason -eq "sqlite-online-backup") {
            if (Test-Path -LiteralPath (Join-Path $Repository ($path -replace "/", "\")) -PathType Leaf) {
                [void]$sqlite.Add($path)
            }
            continue
        }
        if ($null -ne $reason) {
            [void]$excluded.Add([pscustomobject]@{ Path = $path; Reason = $reason; Source = "untracked" })
            continue
        }
        [void]$untrackedSafe.Add($path)
    }

    foreach ($pathValue in @(($dirty + $staged) | Sort-Object -Unique)) {
        $path = Get-NormalizedRelativePath $pathValue
        $reason = Get-ExclusionReason $path
        if ($null -eq $reason) {
            [void]$dirtySafe.Add($path)
        }
        else {
            [void]$excluded.Add([pscustomobject]@{ Path = $path; Reason = $reason; Source = "dirty" })
        }
    }

    $headLines = @(Invoke-GitLines $Repository @("rev-parse", "HEAD"))
    $head = $headLines[0].Trim()
    $branchLines = @(Invoke-GitLines $Repository @("branch", "--show-current"))
    $branch = if ($branchLines.Count -eq 0) { "DETACHED" } else { $branchLines[0].Trim() }
    $status = @(Invoke-GitLines $Repository @("status", "--porcelain=v2", "--branch"))

    $upstream = $null
    $upstreamOid = $null
    $ahead = 0
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $upstreamProbe = @(& git -C $Repository rev-parse --abbrev-ref --symbolic-full-name "@{u}" 2>$null)
        $upstreamExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($upstreamExitCode -eq 0 -and $upstreamProbe.Count -gt 0) {
        $upstream = ([string]$upstreamProbe[0]).Trim()
        $upstreamOid = @(Invoke-GitLines $Repository @("rev-parse", "@{u}"))[0].Trim()
        $previousPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            [void](& git -C $Repository merge-base --is-ancestor "@{u}" HEAD 2>$null)
            $ancestorExitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $previousPreference
        }
        if ($ancestorExitCode -eq 0) {
            $ahead = [int](@(Invoke-GitLines $Repository @("rev-list", "--count", "@{u}..HEAD"))[0])
        }
    }

    return [pscustomobject]@{
        Label = $Label
        Repository = $Repository
        Head = $head
        Branch = $branch
        Upstream = $upstream
        UpstreamOid = $upstreamOid
        Ahead = $ahead
        Status = @($status)
        TrackedSafe = @($trackedSafe | Sort-Object -Unique)
        UntrackedSafe = @($untrackedSafe | Sort-Object -Unique)
        DirtySafe = @($dirtySafe | Sort-Object -Unique)
        Sqlite = @($sqlite | Sort-Object -Unique)
        Deleted = @($deleted | Sort-Object -Unique)
        Excluded = @($excluded | ForEach-Object { $_ })
    }
}

function Get-DataInventory {
    param(
        [Parameter(Mandatory = $true)][string]$Repository,
        [Parameter(Mandatory = $true)][string]$RelativeDirectory
    )

    $directory = Join-Path $Repository ($RelativeDirectory -replace "/", "\")
    if (-not (Test-Path -LiteralPath $directory -PathType Container)) {
        return [pscustomobject]@{ Regular = @(); Sqlite = @(); Excluded = @() }
    }

    $regular = New-Object System.Collections.Generic.List[string]
    $sqlite = New-Object System.Collections.Generic.List[string]
    $excluded = New-Object System.Collections.Generic.List[object]
    foreach ($item in Get-ChildItem -LiteralPath $directory -File -Recurse) {
        $relative = Get-RelativePathUnderRoot $Repository $item.FullName
        $reason = Get-ExclusionReason $relative
        if ($reason -eq "private-key-material") {
            Stop-Preservation "data tree contains private-key-like path: $relative"
        }
        if ($reason -eq "sqlite-online-backup") {
            [void]$sqlite.Add($relative)
        }
        elseif ($null -eq $reason) {
            [void]$regular.Add($relative)
        }
        else {
            [void]$excluded.Add([pscustomobject]@{ Path = $relative; Reason = $reason; Source = "data" })
        }
    }
    return [pscustomobject]@{
        Regular = @($regular | Sort-Object -Unique)
        Sqlite = @($sqlite | Sort-Object -Unique)
        Excluded = @($excluded | ForEach-Object { $_ })
    }
}

function Get-RootRuntimeInventory {
    param([Parameter(Mandatory = $true)][string]$Repository)

    $paths = New-Object System.Collections.Generic.List[string]
    $runtime = Join-Path $Repository "data\runtime"
    if (Test-Path -LiteralPath $runtime -PathType Container) {
        foreach ($item in Get-ChildItem -LiteralPath $runtime -File) {
            if ($item.Extension.ToLowerInvariant() -in @(".json", ".jsonl")) {
                $relative = Get-RelativePathUnderRoot $Repository $item.FullName
                if ($null -eq (Get-ExclusionReason $relative)) {
                    [void]$paths.Add($relative)
                }
            }
        }
        $locks = Join-Path $runtime "alert_locks"
        if (Test-Path -LiteralPath $locks -PathType Container) {
            foreach ($item in Get-ChildItem -LiteralPath $locks -File -Recurse -Filter "*.lock") {
                $relative = Get-RelativePathUnderRoot $Repository $item.FullName
                if ($null -eq (Get-ExclusionReason $relative)) {
                    [void]$paths.Add($relative)
                }
            }
        }
    }

    $observation = Join-Path $Repository "logs\observation"
    if (Test-Path -LiteralPath $observation -PathType Container) {
        foreach ($item in Get-ChildItem -LiteralPath $observation -File -Filter "*.json") {
            $relative = Get-RelativePathUnderRoot $Repository $item.FullName
            if ($null -eq (Get-ExclusionReason $relative)) {
                [void]$paths.Add($relative)
            }
        }
    }
    return @($paths | Sort-Object -Unique)
}

function Get-FileSetBytes {
    param(
        [Parameter(Mandatory = $true)][string]$BasePath,
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][string[]]$RelativePaths
    )
    [int64]$total = 0
    foreach ($relative in $RelativePaths) {
        $item = Get-Item -LiteralPath (Join-Path $BasePath ($relative -replace "/", "\"))
        $total += [int64]$item.Length
    }
    return $total
}

function Get-PathStorageInfo {
    param([Parameter(Mandatory = $true)][string]$Path)

    $fullPath = [IO.Path]::GetFullPath($Path)
    if ($fullPath.StartsWith("\\")) {
        $server = $fullPath.TrimStart("\").Split("\")[0]
        return [pscustomobject]@{
            Kind = "UNC"
            Root = "\\$server"
            Server = $server
            DriveLetter = $null
            DiskNumber = $null
            FileSystem = "UNKNOWN"
            DriveType = "Network"
        }
    }

    $root = [IO.Path]::GetPathRoot($fullPath)
    if ($root -notmatch "^[A-Za-z]:\\$") {
        Stop-Preservation "unsupported destination root: $root"
    }
    $letter = $root.Substring(0, 1).ToUpperInvariant()
    $partition = Get-Partition -DriveLetter $letter -ErrorAction Stop
    $disk = $partition | Get-Disk -ErrorAction Stop
    $volume = Get-Volume -DriveLetter $letter -ErrorAction Stop
    return [pscustomobject]@{
        Kind = "LocalVolume"
        Root = $root
        Server = $null
        DriveLetter = $letter
        DiskNumber = [int]$disk.Number
        FileSystem = [string]$volume.FileSystem
        DriveType = [string]$volume.DriveType
    }
}

function Get-FreeBytes {
    param([Parameter(Mandatory = $true)][string]$Path)

    $fullPath = [IO.Path]::GetFullPath($Path)
    if ($fullPath -notlike "\\*") {
        $letter = [IO.Path]::GetPathRoot($fullPath).Substring(0, 1)
        return [int64](Get-Volume -DriveLetter $letter -ErrorAction Stop).SizeRemaining
    }

    if (-not ("PreservationDiskSpace" -as [type])) {
        Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class PreservationDiskSpace {
    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern bool GetDiskFreeSpaceEx(
        string directoryName,
        out ulong freeBytesAvailable,
        out ulong totalBytes,
        out ulong totalFreeBytes);
}
"@
    }
    [uint64]$available = 0
    [uint64]$total = 0
    [uint64]$free = 0
    if (-not [PreservationDiskSpace]::GetDiskFreeSpaceEx($fullPath, [ref]$available, [ref]$total, [ref]$free)) {
        Stop-Preservation "cannot determine UNC free space"
    }
    return [int64]$available
}

function Assert-ModeStoragePolicy {
    param(
        [Parameter(Mandatory = $true)][string]$SelectedMode,
        [Parameter(Mandatory = $true)]$SourceStorage,
        [Parameter(Mandatory = $true)]$DestinationStorage
    )

    if ($SelectedMode -ne "DisasterResilient") { return }
    if ($DestinationStorage.Kind -eq "UNC") {
        $localNames = @(
            ".", "localhost", "127.0.0.1", "::1", $env:COMPUTERNAME
        ) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
        if ($DestinationStorage.Server -in $localNames) {
            Stop-Preservation "DisasterResilient UNC points to the local computer"
        }
        return
    }
    if ($SourceStorage.Kind -ne "LocalVolume" -or $null -eq $SourceStorage.DiskNumber -or
        $null -eq $DestinationStorage.DiskNumber) {
        Stop-Preservation "cannot prove physical disk separation"
    }
    if ($SourceStorage.DiskNumber -eq $DestinationStorage.DiskNumber) {
        Stop-Preservation "DisasterResilient destination is on the source physical disk"
    }
}

function Save-JsonFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][AllowNull()][AllowEmptyCollection()]$Value,
        [int]$Depth = 10
    )
    if (Test-Path -LiteralPath $Path) {
        Stop-Preservation "refusing to overwrite: $Path"
    }
    $json = ConvertTo-Json -InputObject $Value -Depth $Depth
    [IO.File]::WriteAllText($Path, $json + [Environment]::NewLine, (New-Object Text.UTF8Encoding($false)))
}

function Save-JsonFileAtomic {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][AllowNull()][AllowEmptyCollection()]$Value,
        [int]$Depth = 10
    )
    if (Test-Path -LiteralPath $Path) {
        Stop-Preservation "refusing to overwrite: $Path"
    }
    $temporary = "$Path.incomplete-$([Guid]::NewGuid().ToString('N'))"
    Save-JsonFile $temporary $Value $Depth
    [IO.File]::Move($temporary, $Path)
}

function Get-SharedFileSHA256 {
    param([Parameter(Mandatory = $true)][string]$Path)

    $share = [IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete
    $stream = New-Object IO.FileStream(
        $Path,
        [IO.FileMode]::Open,
        [IO.FileAccess]::Read,
        $share
    )
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        $bytes = $sha.ComputeHash($stream)
        return ([BitConverter]::ToString($bytes)).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $sha.Dispose()
        $stream.Dispose()
    }
}

function Get-FileSetManifestRows {
    param(
        [Parameter(Mandatory = $true)][string]$BasePath,
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][string[]]$RelativePaths
    )

    $rows = New-Object System.Collections.Generic.List[object]
    foreach ($relative in @($RelativePaths | Sort-Object -Unique)) {
        $fullPath = Join-Path $BasePath ($relative -replace "/", "\")
        $item = Get-Item -LiteralPath $fullPath -Force
        if ($item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            Stop-Preservation "non-regular source or destination file: $fullPath"
        }
        [void]$rows.Add([pscustomobject]@{
            RelativePath = $relative
            Length = [int64]$item.Length
            SHA256 = Get-SharedFileSHA256 $fullPath
        })
    }
    return @($rows | ForEach-Object { $_ })
}

function Assert-ManifestsEqual {
    param(
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][object[]]$Expected,
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][object[]]$Actual,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $difference = @(Compare-Object @($Expected) @($Actual) -Property RelativePath, Length, SHA256)
    if ($difference.Count -ne 0) {
        Stop-Preservation "manifest mismatch: $Label"
    }
}

function Copy-VerifiedFileSet {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$SourceRoot,
        [Parameter(Mandatory = $true)][string]$DestinationRoot,
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][string[]]$RelativePaths,
        [Parameter(Mandatory = $true)][string]$ManifestRoot
    )

    if (-not (Test-Path -LiteralPath $DestinationRoot)) {
        [void](New-Item -ItemType Directory -Path $DestinationRoot)
    }
    $before = @(Get-FileSetManifestRows $SourceRoot $RelativePaths)
    Save-JsonFile (Join-Path $ManifestRoot "${Name}_source_before.json") @($before)

    foreach ($relative in @($RelativePaths | Sort-Object -Unique)) {
        $source = Join-Path $SourceRoot ($relative -replace "/", "\")
        $destination = Join-Path $DestinationRoot ($relative -replace "/", "\")
        if (Test-Path -LiteralPath $destination) {
            Stop-Preservation "refusing to overwrite copied file: $destination"
        }
        $parent = Split-Path -Parent $destination
        if (-not (Test-Path -LiteralPath $parent)) {
            [void](New-Item -ItemType Directory -Path $parent -Force)
        }
        Copy-Item -LiteralPath $source -Destination $destination
    }

    $after = @(Get-FileSetManifestRows $SourceRoot $RelativePaths)
    $destinationRows = @(Get-FileSetManifestRows $DestinationRoot $RelativePaths)
    Save-JsonFile (Join-Path $ManifestRoot "${Name}_source_after.json") @($after)
    Save-JsonFile (Join-Path $ManifestRoot "${Name}_destination.json") @($destinationRows)
    Assert-ManifestsEqual $before $after "$Name source changed during copy"
    Assert-ManifestsEqual $before $destinationRows "$Name destination"

    [int64]$byteTotal = 0
    if ($before.Count -gt 0) {
        $byteTotal = [int64](($before | Measure-Object Length -Sum).Sum)
    }
    return [pscustomobject]@{
        Name = $Name
        Files = $before.Count
        Bytes = $byteTotal
    }
}

function Get-LiteralPathspecs {
    param([Parameter(Mandatory = $true)][AllowEmptyCollection()][string[]]$RelativePaths)
    return @($RelativePaths | Sort-Object -Unique | ForEach-Object { ":(literal)$_" })
}

function Assert-GitDiffContainsNoSecrets {
    param(
        [Parameter(Mandatory = $true)][string]$Repository,
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][string[]]$RelativePaths,
        [string]$Range = "HEAD",
        [switch]$Cached,
        [switch]$WorktreeOnly,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $textPaths = @($RelativePaths | Where-Object { Test-TextCandidate $_ })
    if ($textPaths.Count -eq 0) { return }
    $arguments = @("diff", "--no-ext-diff", "--unified=0", "--text")
    if ($Cached) { $arguments += "--cached" }
    if (-not $WorktreeOnly) { $arguments += $Range }
    $arguments += "--"
    $arguments += Get-LiteralPathspecs $textPaths
    $lines = @(Invoke-GitLines $Repository $arguments)
    Assert-LinesContainNoSecrets $lines $Label
}

function Write-GitPatch {
    param(
        [Parameter(Mandatory = $true)][string]$Repository,
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][string[]]$RelativePaths,
        [Parameter(Mandatory = $true)][string]$Destination,
        [ValidateSet("Combined", "Index", "Worktree")][string]$Kind
    )

    if (Test-Path -LiteralPath $Destination) {
        Stop-Preservation "refusing to overwrite patch: $Destination"
    }
    if ($RelativePaths.Count -eq 0) {
        [IO.File]::WriteAllBytes($Destination, [byte[]]@())
        return
    }
    $arguments = @(
        "-c", "core.quotepath=false", "-C", $Repository, "diff", "--binary",
        "--full-index", "--output=$Destination"
    )
    if ($Kind -eq "Index") {
        $arguments += "--cached"
        $arguments += "HEAD"
    }
    elseif ($Kind -eq "Combined") {
        $arguments += "HEAD"
    }
    $arguments += "--"
    $arguments += Get-LiteralPathspecs $RelativePaths
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & git @arguments 2>$null
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($exitCode -ne 0) {
        Stop-Preservation "failed to write $Kind patch for $Repository"
    }
}

function Export-SafeCommitSeries {
    param(
        [Parameter(Mandatory = $true)]$Inventory,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    if ([string]::IsNullOrWhiteSpace([string]$Inventory.Upstream) -or $Inventory.Ahead -le 0) {
        return [pscustomobject]@{ Exported = $false; Count = 0; Reason = "no-linear-upstream-range" }
    }
    $range = "$($Inventory.Upstream)..HEAD"
    $changed = @(Invoke-GitLines $Inventory.Repository @("diff", "--name-only", $range, "--") | Where-Object { $_ })
    foreach ($pathValue in $changed) {
        $reason = Get-ExclusionReason $pathValue
        if ($null -ne $reason) {
            return [pscustomobject]@{ Exported = $false; Count = 0; Reason = "range-touches-excluded-path:$reason" }
        }
    }
    Assert-GitDiffContainsNoSecrets $Inventory.Repository $changed -Range $range -Label "$($Inventory.Label)-commit-range"
    [void](New-Item -ItemType Directory -Path $Destination)
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & git -C $Inventory.Repository format-patch --binary --full-index --output-directory $Destination $range 2>$null | Out-Null
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($exitCode -ne 0) {
        Stop-Preservation "format-patch failed for $($Inventory.Label)"
    }
    $patches = @(Get-ChildItem -LiteralPath $Destination -File -Filter "*.patch")
    if ($patches.Count -ne $Inventory.Ahead) {
        Stop-Preservation "commit patch count mismatch for $($Inventory.Label)"
    }
    foreach ($patch in $patches) {
        $lineNumber = 0
        Get-Content -LiteralPath $patch.FullName -Encoding UTF8 -ReadCount 1 | ForEach-Object {
            $lineNumber += 1
            if ([string]$_ -match "^diff --git a/\.env(?:\..*)? b/\.env(?:\..*)?$") {
                Stop-Preservation "environment file appeared in commit patch $($patch.Name)"
            }
        }
    }
    return [pscustomobject]@{ Exported = $true; Count = $patches.Count; Reason = $null }
}

function Get-DatabaseSourceState {
    param([Parameter(Mandatory = $true)][AllowEmptyCollection()][object[]]$Jobs)

    $rows = New-Object System.Collections.Generic.List[object]
    foreach ($job in $Jobs) {
        foreach ($path in @([string]$job.Source, ([string]$job.Source + "-wal"))) {
            if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { continue }
            $item = Get-Item -LiteralPath $path
            [void]$rows.Add([pscustomobject]@{
                Path = $path
                Length = [int64]$item.Length
                SHA256 = Get-SharedFileSHA256 $path
            })
        }
    }
    return @($rows | Sort-Object Path)
}

function Invoke-SqliteOnlineBackup {
    param(
        [Parameter(Mandatory = $true)][string]$Interpreter,
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][object[]]$Jobs,
        [Parameter(Mandatory = $true)][string]$MetadataRoot
    )

    $jobsPath = Join-Path $MetadataRoot "sqlite_jobs.json"
    $reportPath = Join-Path $MetadataRoot "sqlite_backup_report.json"
    Save-JsonFile $jobsPath @($Jobs)
    $before = @(Get-DatabaseSourceState $Jobs)

    $env:KOTEIKA_PRESERVATION_SQLITE_JOBS = $jobsPath
    $env:KOTEIKA_PRESERVATION_SQLITE_REPORT = $reportPath
    try {
        @'
import json
import os
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

jobs_path = Path(os.environ["KOTEIKA_PRESERVATION_SQLITE_JOBS"])
report_path = Path(os.environ["KOTEIKA_PRESERVATION_SQLITE_REPORT"])
jobs = json.loads(jobs_path.read_text(encoding="utf-8"))

def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'

def table_counts(connection: sqlite3.Connection) -> dict[str, int]:
    names = [
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]
    return {
        name: connection.execute(
            f"SELECT COUNT(*) FROM {quote_identifier(name)}"
        ).fetchone()[0]
        for name in names
    }

report = {"sqlite_version": sqlite3.sqlite_version, "databases": []}
for job in jobs:
    source_path = Path(job["Source"]).resolve()
    destination_path = Path(job["Destination"]).resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists():
        raise RuntimeError(f"destination exists: {destination_path}")

    source_size = source_path.stat().st_size
    if source_size == 0:
        shutil.copyfile(source_path, destination_path)
        report["databases"].append(
            {
                "label": job["Label"],
                "source": str(source_path),
                "destination": str(destination_path),
                "kind": "zero-byte-copy",
                "source_size": 0,
                "destination_size": 0,
            }
        )
        continue

    source_uri = source_path.as_uri() + "?mode=ro"
    with closing(sqlite3.connect(source_uri, uri=True, timeout=60)) as source:
        source.execute("PRAGMA query_only=ON")
        data_version_before = source.execute("PRAGMA data_version").fetchone()[0]
        with closing(sqlite3.connect(destination_path, timeout=60)) as destination:
            source.backup(destination, pages=4096, sleep=0.05)
            destination.commit()
            integrity = [row[0] for row in destination.execute("PRAGMA integrity_check")]
            if integrity != ["ok"]:
                raise RuntimeError(f"integrity_check failed for {job['Label']}")
            source_counts = table_counts(source)
            destination_counts = table_counts(destination)
            if source_counts != destination_counts:
                raise RuntimeError(f"table counts differ for {job['Label']}")
        data_version_after = source.execute("PRAGMA data_version").fetchone()[0]
        if data_version_before != data_version_after:
            raise RuntimeError(f"source changed during backup for {job['Label']}")

    report["databases"].append(
        {
            "label": job["Label"],
            "source": str(source_path),
            "destination": str(destination_path),
            "kind": "sqlite-online-backup",
            "source_size": source_size,
            "destination_size": destination_path.stat().st_size,
            "integrity_check": "ok",
            "table_counts": destination_counts,
        }
    )

report_path.write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
'@ | & $Interpreter -
        if ($LASTEXITCODE -ne 0) {
            Stop-Preservation "SQLite online backup failed"
        }
    }
    finally {
        Remove-Item Env:KOTEIKA_PRESERVATION_SQLITE_JOBS -ErrorAction SilentlyContinue
        Remove-Item Env:KOTEIKA_PRESERVATION_SQLITE_REPORT -ErrorAction SilentlyContinue
    }

    $after = @(Get-DatabaseSourceState $Jobs)
    Assert-ManifestsEqual $before $after "SQLite DB/WAL source changed"
    if (-not (Test-Path -LiteralPath $reportPath -PathType Leaf)) {
        Stop-Preservation "SQLite report missing"
    }
    return Get-Content -LiteralPath $reportPath -Raw -Encoding UTF8 | ConvertFrom-Json
}

function Get-BackupPayloadRows {
    param(
        [Parameter(Mandatory = $true)][string]$BackupRoot,
        [Parameter(Mandatory = $true)][string[]]$ExcludedRelativePaths
    )

    $rows = New-Object System.Collections.Generic.List[object]
    foreach ($item in Get-ChildItem -LiteralPath $BackupRoot -File -Recurse) {
        $relative = Get-RelativePathUnderRoot $BackupRoot $item.FullName
        if ($relative -in $ExcludedRelativePaths) { continue }
        [void]$rows.Add([pscustomobject]@{
            RelativePath = $relative
            Length = [int64]$item.Length
            SHA256 = Get-SharedFileSHA256 $item.FullName
        })
    }
    return @($rows | Sort-Object RelativePath)
}

function Assert-NoForbiddenBackupPaths {
    param([Parameter(Mandatory = $true)][string]$BackupRoot)

    foreach ($item in Get-ChildItem -LiteralPath $BackupRoot -Force -Recurse) {
        $relative = Get-RelativePathUnderRoot $BackupRoot $item.FullName
        $parts = @($relative -split "/")
        $leaf = $parts[$parts.Count - 1]
        if ($leaf -match "(?i)^\.env(?:\..*)?$" -or
            $leaf -match "(?i)\.db-(?:wal|shm)$" -or
            $relative -match "(?i)(^|/)logs/system\.log$" -or
            $relative -match "(?i)(^|/)data/runtime/bot_runtime\.lock$") {
            Stop-Preservation "forbidden path entered backup: $relative"
        }
        foreach ($part in $parts) {
            if ($part -eq ".git" -or $part -match "(?i)^\.venv(?:_|$)" -or
                $part -match "(?i)^(?:__pycache__|\.pytest_cache|\.mypy_cache|\.ruff_cache|cache|caches)$") {
                Stop-Preservation "forbidden directory entered backup: $relative"
            }
        }
    }
}

function Assert-GitInventoryStable {
    param(
        [Parameter(Mandatory = $true)]$Before,
        [Parameter(Mandatory = $true)]$After
    )

    if ($Before.Head -ne $After.Head -or $Before.Branch -ne $After.Branch) {
        Stop-Preservation "$($Before.Label) Git tip changed during backup"
    }
    foreach ($property in @("Status", "TrackedSafe", "UntrackedSafe", "DirtySafe", "Sqlite", "Deleted")) {
        $difference = @(Compare-Object @($Before.$property) @($After.$property))
        if ($difference.Count -ne 0) {
            Stop-Preservation "$($Before.Label) Git inventory changed during backup: $property"
        }
    }
}

if ([string]::IsNullOrWhiteSpace($RootPath)) {
    $RootPath = Join-Path $PSScriptRoot "..\.."
}
$RootPath = (Resolve-Path -LiteralPath $RootPath).Path

if ([string]::IsNullOrWhiteSpace($MexcPath)) {
    $MexcPath = Join-Path $RootPath ".claude\worktrees\codex-project-review-04581e"
}
if (-not (Test-Path -LiteralPath $MexcPath -PathType Container)) {
    Stop-Preservation "MEXC worktree is missing: $MexcPath"
}
$MexcPath = (Resolve-Path -LiteralPath $MexcPath).Path

if ([string]::IsNullOrWhiteSpace($PythonPath)) {
    $venvPython = Join-Path $RootPath ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
        $PythonPath = $venvPython
    }
    else {
        $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
        if ($null -eq $pythonCommand) {
            Stop-Preservation "Python interpreter is unavailable"
        }
        $PythonPath = $pythonCommand.Source
    }
}
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    Stop-Preservation "Python interpreter is missing: $PythonPath"
}

$BackupBaseFull = [IO.Path]::GetFullPath($BackupBase)
if ((Test-PathInside $BackupBaseFull $RootPath) -or (Test-PathInside $BackupBaseFull $MexcPath)) {
    Stop-Preservation "destination is inside a source worktree"
}
Assert-DestinationAncestorsSafe $BackupBaseFull

$rootInventory = Get-GitInventory $RootPath "root"
$mexcInventory = Get-GitInventory $MexcPath "mexc"
$rootRuntime = @(Get-RootRuntimeInventory $RootPath)
$mexcData = Get-DataInventory $MexcPath "data"

Assert-FilesContainNoSecrets $RootPath @($rootInventory.TrackedSafe + $rootInventory.UntrackedSafe + $rootRuntime) "root"
Assert-FilesContainNoSecrets $MexcPath @($mexcInventory.TrackedSafe + $mexcInventory.UntrackedSafe) "mexc"
Assert-GitDiffContainsNoSecrets $RootPath $rootInventory.DirtySafe -Range "HEAD" -Label "root-dirty"
Assert-GitDiffContainsNoSecrets $MexcPath $mexcInventory.DirtySafe -Range "HEAD" -Label "mexc-dirty"

$projectProcesses = @(
    Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match "^pythonw?\.exe$" -and
        (
            ([string]$_.CommandLine).IndexOf($RootPath, [StringComparison]::OrdinalIgnoreCase) -ge 0 -or
            ([string]$_.CommandLine).IndexOf($MexcPath, [StringComparison]::OrdinalIgnoreCase) -ge 0
        )
    }
)
if ($projectProcesses.Count -ne 0) {
    Stop-Preservation "project-related Python process is active"
}

& $PythonPath -c "import sqlite3; assert hasattr(sqlite3.Connection, 'backup')" 2>$null
if ($LASTEXITCODE -ne 0) {
    Stop-Preservation "Python SQLite online backup API is unavailable"
}

$rootStorage = Get-PathStorageInfo $RootPath
$destinationStorage = Get-PathStorageInfo $BackupBaseFull
Assert-ModeStoragePolicy $Mode $rootStorage $destinationStorage

$rootDbPaths = @($rootInventory.Sqlite)
$rootRuntimeDirectory = Join-Path $RootPath "data\runtime"
if (Test-Path -LiteralPath $rootRuntimeDirectory -PathType Container) {
    $rootDbPaths += @(
        Get-ChildItem -LiteralPath $rootRuntimeDirectory -File -Recurse -Filter "*.db" |
        ForEach-Object { Get-RelativePathUnderRoot $RootPath $_.FullName }
    )
}
$rootDbPaths = @($rootDbPaths | Sort-Object -Unique)
$mexcDbPaths = @($mexcInventory.Sqlite + $mexcData.Sqlite | Sort-Object -Unique)

foreach ($dbSource in @(
    @($rootDbPaths | ForEach-Object { Join-Path $RootPath ($_ -replace "/", "\") }) +
    @($mexcDbPaths | ForEach-Object { Join-Path $MexcPath ($_ -replace "/", "\") })
)) {
    $dbItem = Get-Item -LiteralPath $dbSource
    $walPath = $dbSource + "-wal"
    $shmPath = $dbSource + "-shm"
    if ($dbItem.Length -eq 0 -and
        (((Test-Path -LiteralPath $walPath -PathType Leaf) -and (Get-Item -LiteralPath $walPath).Length -gt 0) -or
         ((Test-Path -LiteralPath $shmPath -PathType Leaf) -and (Get-Item -LiteralPath $shmPath).Length -gt 0))) {
        Stop-Preservation "zero-byte SQLite database has a non-empty sidecar: $dbSource"
    }
}

$allSets = @(
    [pscustomobject]@{ Base = $RootPath; Paths = @($rootInventory.TrackedSafe) },
    [pscustomobject]@{ Base = $RootPath; Paths = @($rootInventory.UntrackedSafe) },
    [pscustomobject]@{ Base = $RootPath; Paths = @($rootRuntime) },
    [pscustomobject]@{ Base = $MexcPath; Paths = @($mexcInventory.TrackedSafe) },
    [pscustomobject]@{ Base = $MexcPath; Paths = @($mexcInventory.UntrackedSafe) },
    [pscustomobject]@{ Base = $MexcPath; Paths = @($mexcData.Regular) },
    [pscustomobject]@{ Base = $RootPath; Paths = @($rootDbPaths) },
    [pscustomobject]@{ Base = $MexcPath; Paths = @($mexcDbPaths) }
)
[int64]$estimatedPayload = 0
foreach ($set in $allSets) {
    $estimatedPayload += Get-FileSetBytes $set.Base $set.Paths
}
foreach ($dbSource in @(
    @($rootDbPaths | ForEach-Object { Join-Path $RootPath ($_ -replace "/", "\") }) +
    @($mexcDbPaths | ForEach-Object { Join-Path $MexcPath ($_ -replace "/", "\") })
)) {
    $walPath = $dbSource + "-wal"
    if (Test-Path -LiteralPath $walPath -PathType Leaf) {
        $estimatedPayload += [int64](Get-Item -LiteralPath $walPath).Length
    }
}
[int64]$requiredBytes = [int64][Math]::Ceiling(($estimatedPayload * 1.25) + 512MB)
[int64]$freeBytes = Get-FreeBytes $BackupBaseFull
if ($freeBytes -lt $requiredBytes) {
    Stop-Preservation "insufficient destination space required=$requiredBytes available=$freeBytes"
}

if ($destinationStorage.FileSystem -eq "FAT32") {
    [int64]$fat32Maximum = 4294967295
    foreach ($set in $allSets) {
        foreach ($relative in $set.Paths) {
            $length = (Get-Item -LiteralPath (Join-Path $set.Base ($relative -replace "/", "\"))).Length
            if ($length -gt $fat32Maximum) {
                Stop-Preservation "FAT32 cannot store source file: $relative"
            }
        }
    }
    foreach ($dbSource in @(
        @($rootDbPaths | ForEach-Object { Join-Path $RootPath ($_ -replace "/", "\") }) +
        @($mexcDbPaths | ForEach-Object { Join-Path $MexcPath ($_ -replace "/", "\") })
    )) {
        [int64]$logicalUpperBound = (Get-Item -LiteralPath $dbSource).Length
        if (Test-Path -LiteralPath ($dbSource + "-wal") -PathType Leaf) {
            $logicalUpperBound += [int64](Get-Item -LiteralPath ($dbSource + "-wal")).Length
        }
        if ($logicalUpperBound -gt $fat32Maximum) {
            Stop-Preservation "FAT32 may not fit SQLite backup: $dbSource"
        }
    }
}

$preflight = [ordered]@{
    SchemaVersion = 1
    Mode = $Mode
    PreflightOnly = [bool]$PreflightOnly
    Root = [ordered]@{
        Path = $RootPath
        Head = $rootInventory.Head
        Branch = $rootInventory.Branch
        TrackedFiles = $rootInventory.TrackedSafe.Count
        UntrackedFiles = $rootInventory.UntrackedSafe.Count
        DirtyFiles = $rootInventory.DirtySafe.Count
        SqliteFiles = $rootDbPaths.Count
    }
    Mexc = [ordered]@{
        Path = $MexcPath
        Head = $mexcInventory.Head
        Branch = $mexcInventory.Branch
        TrackedFiles = $mexcInventory.TrackedSafe.Count
        UntrackedFiles = $mexcInventory.UntrackedSafe.Count
        DirtyFiles = $mexcInventory.DirtySafe.Count
        SqliteFiles = $mexcDbPaths.Count
    }
    Destination = [ordered]@{
        Base = $BackupBaseFull
        Kind = $destinationStorage.Kind
        DiskNumber = $destinationStorage.DiskNumber
        FileSystem = $destinationStorage.FileSystem
        FreeBytes = $freeBytes
        RequiredBytes = $requiredBytes
    }
    EstimatedPayloadBytes = $estimatedPayload
}

if ($PreflightOnly) {
    [Console]::Out.WriteLine((ConvertTo-Json -InputObject $preflight -Depth 8))
    exit 0
}

$runId = "{0}_{1}" -f (Get-Date -Format "yyyyMMdd_HHmmss"), ([Guid]::NewGuid().ToString("N").Substring(0, 8))
$backupRoot = Join-Path $BackupBaseFull "koteika_preservation_$runId"
if (Test-Path -LiteralPath $backupRoot) {
    Stop-Preservation "unique run directory already exists"
}

[void](New-Item -ItemType Directory -Path $backupRoot)
$metadataRoot = Join-Path $backupRoot "metadata"
$manifestRoot = Join-Path $backupRoot "manifests"
$patchRoot = Join-Path $backupRoot "patches"
foreach ($directory in @($metadataRoot, $manifestRoot, $patchRoot)) {
    [void](New-Item -ItemType Directory -Path $directory)
}

Save-JsonFile (Join-Path $metadataRoot "preflight.json") $preflight
Save-JsonFile (Join-Path $metadataRoot "root_git.json") $rootInventory
Save-JsonFile (Join-Path $metadataRoot "mexc_git.json") $mexcInventory

$copyResults = @(
    Copy-VerifiedFileSet "root_tracked" $RootPath (Join-Path $backupRoot "snapshot\root_tracked") $rootInventory.TrackedSafe $manifestRoot
    Copy-VerifiedFileSet "root_untracked" $RootPath (Join-Path $backupRoot "snapshot\root_untracked") $rootInventory.UntrackedSafe $manifestRoot
    Copy-VerifiedFileSet "root_runtime" $RootPath (Join-Path $backupRoot "runtime\root") $rootRuntime $manifestRoot
    Copy-VerifiedFileSet "mexc_tracked" $MexcPath (Join-Path $backupRoot "snapshot\mexc_tracked") $mexcInventory.TrackedSafe $manifestRoot
    Copy-VerifiedFileSet "mexc_untracked" $MexcPath (Join-Path $backupRoot "snapshot\mexc_untracked") $mexcInventory.UntrackedSafe $manifestRoot
    Copy-VerifiedFileSet "mexc_data" $MexcPath (Join-Path $backupRoot "data\mexc") $mexcData.Regular $manifestRoot
)

$rootPatchDirectory = Join-Path $patchRoot "root"
$mexcPatchDirectory = Join-Path $patchRoot "mexc"
[void](New-Item -ItemType Directory -Path $rootPatchDirectory)
[void](New-Item -ItemType Directory -Path $mexcPatchDirectory)
Write-GitPatch $RootPath $rootInventory.DirtySafe (Join-Path $rootPatchDirectory "combined_HEAD.patch") "Combined"
Write-GitPatch $RootPath $rootInventory.DirtySafe (Join-Path $rootPatchDirectory "index.patch") "Index"
Write-GitPatch $RootPath $rootInventory.DirtySafe (Join-Path $rootPatchDirectory "worktree.patch") "Worktree"
Write-GitPatch $MexcPath $mexcInventory.DirtySafe (Join-Path $mexcPatchDirectory "combined_HEAD.patch") "Combined"
Write-GitPatch $MexcPath $mexcInventory.DirtySafe (Join-Path $mexcPatchDirectory "index.patch") "Index"
Write-GitPatch $MexcPath $mexcInventory.DirtySafe (Join-Path $mexcPatchDirectory "worktree.patch") "Worktree"

$rootSeries = Export-SafeCommitSeries $rootInventory (Join-Path $rootPatchDirectory "commit_series")
$mexcSeries = Export-SafeCommitSeries $mexcInventory (Join-Path $mexcPatchDirectory "commit_series")
Save-JsonFile (Join-Path $metadataRoot "commit_series.json") ([ordered]@{ Root = $rootSeries; Mexc = $mexcSeries })

$dbJobs = New-Object System.Collections.Generic.List[object]
foreach ($relative in $rootDbPaths) {
    [void]$dbJobs.Add([pscustomobject]@{
        Label = "root:$relative"
        Source = Join-Path $RootPath ($relative -replace "/", "\")
        Destination = Join-Path $backupRoot ("sqlite\root\" + ($relative -replace "/", "\"))
    })
}
foreach ($relative in $mexcDbPaths) {
    [void]$dbJobs.Add([pscustomobject]@{
        Label = "mexc:$relative"
        Source = Join-Path $MexcPath ($relative -replace "/", "\")
        Destination = Join-Path $backupRoot ("sqlite\mexc\" + ($relative -replace "/", "\"))
    })
}
$destinationCollisions = @(
    $dbJobs |
    Group-Object { ([string]$_.Destination).ToLowerInvariant() } |
    Where-Object { $_.Count -gt 1 }
)
if ($destinationCollisions.Count -ne 0) {
    Stop-Preservation "SQLite destination collision"
}
$sqliteReport = Invoke-SqliteOnlineBackup $PythonPath @($dbJobs | ForEach-Object { $_ }) $metadataRoot

Assert-NoForbiddenBackupPaths $backupRoot

$finalManifestRelative = "MANIFEST_SHA256.json"
$finalSidecarRelative = "MANIFEST_SHA256.json.sha256"
$checkpointMarkerRelative = "CHECKPOINT_VERIFIED.json"
$verifiedMarkerRelative = "VERIFIED_OK.json"
$finalExclusions = @(
    $finalManifestRelative,
    $finalSidecarRelative,
    $checkpointMarkerRelative,
    $verifiedMarkerRelative
)
$payloadRows = @(Get-BackupPayloadRows $backupRoot $finalExclusions)
$finalManifestPath = Join-Path $backupRoot $finalManifestRelative
Save-JsonFile $finalManifestPath @($payloadRows)
$manifestHash = Get-SharedFileSHA256 $finalManifestPath
$sidecarPath = Join-Path $backupRoot $finalSidecarRelative
[IO.File]::WriteAllText($sidecarPath, "$manifestHash  $finalManifestRelative`r`n", [Text.Encoding]::ASCII)

$verifiedRows = @(Get-BackupPayloadRows $backupRoot $finalExclusions)
Assert-ManifestsEqual $payloadRows $verifiedRows "final destination re-read"
$sidecarHash = ((Get-Content -LiteralPath $sidecarPath -Encoding ASCII -Raw) -split "\s+")[0]
if ($sidecarHash -ne (Get-SharedFileSHA256 $finalManifestPath)) {
    Stop-Preservation "final manifest sidecar mismatch"
}
Assert-NoForbiddenBackupPaths $backupRoot

$rootInventoryFinal = Get-GitInventory $RootPath "root"
$mexcInventoryFinal = Get-GitInventory $MexcPath "mexc"
Assert-GitInventoryStable $rootInventory $rootInventoryFinal
Assert-GitInventoryStable $mexcInventory $mexcInventoryFinal

$destinationStorageFinal = Get-PathStorageInfo $BackupBaseFull
Assert-ModeStoragePolicy $Mode $rootStorage $destinationStorageFinal
if ($destinationStorage.Kind -ne $destinationStorageFinal.Kind -or
    $destinationStorage.DiskNumber -ne $destinationStorageFinal.DiskNumber -or
    $destinationStorage.FileSystem -ne $destinationStorageFinal.FileSystem) {
    Stop-Preservation "destination storage identity changed during backup"
}

if ($destinationStorage.FileSystem -eq "FAT32") {
    $oversized = @(Get-ChildItem -LiteralPath $backupRoot -File -Recurse | Where-Object { $_.Length -gt 4294967295 })
    if ($oversized.Count -ne 0) {
        Stop-Preservation "FAT32 payload contains an oversized file"
    }
}

$receipt = [ordered]@{
    SchemaVersion = 1
    Status = "verified"
    Mode = $Mode
    CreatedAt = (Get-Date).ToString("o")
    BackupRoot = $backupRoot
    RootHead = $rootInventory.Head
    MexcHead = $mexcInventory.Head
    SourceDiskNumber = $rootStorage.DiskNumber
    DestinationKind = $destinationStorage.Kind
    DestinationDiskNumber = $destinationStorage.DiskNumber
    DestinationFileSystem = $destinationStorage.FileSystem
    PayloadFiles = $payloadRows.Count
    PayloadBytes = [int64](($payloadRows | Measure-Object Length -Sum).Sum)
    ManifestSHA256 = $manifestHash
    CopyResults = @($copyResults)
    SQLiteDatabases = @($sqliteReport.databases).Count
    CommitSeries = [ordered]@{ Root = $rootSeries; Mexc = $mexcSeries }
    Exclusions = [ordered]@{
        Root = @($rootInventory.Excluded)
        Mexc = @($mexcInventory.Excluded + $mexcData.Excluded)
    }
}

$markerPath = if ($Mode -eq "DisasterResilient") {
    Join-Path $backupRoot $verifiedMarkerRelative
}
else {
    Join-Path $backupRoot $checkpointMarkerRelative
}
Save-JsonFileAtomic $markerPath $receipt

[Console]::Out.WriteLine((ConvertTo-Json -InputObject ([ordered]@{
    Status = "verified"
    Mode = $Mode
    BackupRoot = $backupRoot
    Marker = $markerPath
    ManifestSHA256 = $manifestHash
}) -Depth 5))
