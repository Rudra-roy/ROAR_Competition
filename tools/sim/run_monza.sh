#!/usr/bin/env bash
# Start the ROAR Monza CARLA 0.9.12 server (a Windows build) on Linux under GE-Proton 9. Extra args are passed to the
# server (e.g. -RenderOffScreen -carla-rpc-port=2003 -carla-streaming-port=2004). Paths can be overridden:
#   ROAR_MONZA   extracted map folder (contains CarlaUE4.exe)   default ~/Documents/roar_racing/Monza_V1.1/Monza
#   ROAR_PROTON  GE-Proton 9 folder                             default ~/tools/GE-Proton9-27
#   ROAR_PFX     Proton prefix                                  default ~/tools/pfx
# -dx12 is required (the DX11 path asserts); never pass -quality-level=Low (missing materials crash the map).
# Proton needs system python >= 3.9, so run this outside the Python 3.8 venv.
MONZA=${ROAR_MONZA:-$HOME/Documents/roar_racing/Monza_V1.1/Monza}
PROTON=${ROAR_PROTON:-$HOME/tools/GE-Proton9-27}
export STEAM_COMPAT_DATA_PATH=${ROAR_PFX:-$HOME/tools/pfx}
export STEAM_COMPAT_CLIENT_INSTALL_PATH=$HOME/.steam
export PROTON_ENABLE_NVAPI=1 DXVK_ENABLE_NVAPI=1 SteamAppId=0 SteamGameId=0
cd "$MONZA" || exit 1
exec "$PROTON/proton" run "$MONZA/CarlaUE4/Binaries/Win64/CarlaUE4.exe" CarlaUE4 -dx12 -windowed -ResX=1280 -ResY=720 "$@"
