# Get the current directory
$currentDirectory = Get-Location

# Get all files or directories in the current directory
$files = Get-ChildItem $currentDirectory

Write-Output $files

$wordDictionary = @{
    "aslan" = "lion"
    "esek" = "donkey"
    "inek" = "cow"
    "kedi" = "cat"
    "kopek" = "dog"
    "koyun" = "sheep"
    "kurbaga" = "frog"
    "maymun" = "monkey"
    "tavuk" = "chicken"
}

# Loop through each file/directory
foreach ($file in $files) {
    
    # check if the file is a directory, if so, loop through each file in that directory
    if ($file.PSIsContainer) {

        # Rename container with translation if in turkish
        foreach ($key in $wordDictionary.Keys) {

            # Check if the filename contains the word
            if ($file.Name -match $key) {
                # Replace "kus" with "bird" in the filename
                $newName = $file.Name -replace $key, $wordDictionary[$key]
                
                # Rename the file
                Rename-Item -Path $file.FullName -NewName $newName
            }
        }    


        # Get all files inside this directory again, so that we can loop through the files in the dir too.
        $filesindir = Get-ChildItem $file.FullName
        
        # go through files in this directory
        foreach ($fileInDir in $filesindir) {
            
            #Check for each word in dictionary
            foreach ($key in $wordDictionary.Keys) {

                # Check if the filename contains the word
                if ($fileInDir.Name -match $key) {
                    # Replace "kus" with "bird" in the filename
                    $newName = $fileInDir.Name -replace $key, $wordDictionary[$key]
                    
                    # Rename the file
                    Rename-Item -Path $fileInDir.FullName -NewName $newName
                }
            }
        }
    }
}

Read-Host -Prompt "Press Enter to exit"
