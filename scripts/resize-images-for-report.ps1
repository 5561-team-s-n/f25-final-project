# this script crunches down the size of images in our final report,
# so the final pdf isn't so large. 
# to be run on windows (i did most of my training there, so whatever. would prefer
# a bash script but pwsh works 

$src = # "YOUR_PATH_TO_FIGURES"
$src = "K:\projects\skool\ee5561-final-project\tex\figs"
$dst = Join-Path (Split-Path $src -Parent) "figs_optimized"
New-Item -ItemType Directory -Force -Path $dst | Out-Null

$maxDimPng = 1000
$maxDimJpg = 1000
$jpgQ = 90 # smaller = lower quality 
$pngQ = 90 # smaller = lower quality (?)
$strip = $true # get rid of metadata
$outputAllAsJpg = $true # if true, write everything (including pngs) as jpgs

# for all the pngs
Get-ChildItem $src -Recurse -File -Include *.png | ForEach-Object {
  $rel = $_.FullName.Substring($src.Length).TrimStart("\")
  $outRel = if ($outputAllAsJpg) { [IO.Path]::ChangeExtension($rel, ".jpg") } else { $rel }
  $out = Join-Path $dst $outRel
  New-Item -ItemType Directory -Force -Path (Split-Path $out -Parent) | Out-Null

  $args = @($_.FullName,
    "-auto-orient"
  )
  if ($strip) { $args += "-strip" }

  if ($outputAllAsJpg) {
    $args += @(
      "-resize", "$maxDimPng`x$maxDimPng`>",
      "-background", "white",
      "-alpha", "remove",
      "-alpha", "off",
      "-sampling-factor", "4:2:0",
      "-interlace", "Plane",
      "-quality", "$jpgQ",
      $out
    )
  } else {
    $args += @(
      "-resize", "$maxDimPng`x$maxDimPng`>",
      "-define", "png:compression-level=9",
      "-quality", "$pngQ",
      $out
    )
  }

  magick @args
}

# for all the jpgs
Get-ChildItem $src -Recurse -File -Include *.jpg, *.jpeg | ForEach-Object {
  $rel = $_.FullName.Substring($src.Length).TrimStart("\")
  $outRel = if ($outputAllAsJpg) { [IO.Path]::ChangeExtension($rel, ".jpg") } else { $rel }
  $out = Join-Path $dst $outRel
  New-Item -ItemType Directory -Force -Path (Split-Path $out -Parent) | Out-Null

  $args = @($_.FullName,
    "-auto-orient"
  )
  if ($strip) { $args += "-strip" }
  $args += @(
    "-resize", "$maxDimJpg`x$maxDimJpg`>",
    "-sampling-factor", "4:2:0",
    "-interlace", "Plane",
    "-quality", "$jpgQ",
    $out
  )
  magick @args
}
