# Get the current directory
$currentDirectory = Get-Location

# Get all files in the current directory
$files = Get-ChildItem $currentDirectory

Write-Output $files

# Loop through each file
foreach ($file in $files) {
    # Check if the filename contains the word "kus"

    if ($file.Name -match "kus") {
        # Replace "kus" with "bird" in the filename
        $newName = $file.Name -replace "kus", "bird"
        
        # Rename the file
        Rename-Item -Path $file.FullName -NewName $newName
    }
}

Read-Host -Prompt "Press Enter to exit"
