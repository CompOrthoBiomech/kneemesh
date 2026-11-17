#!/bin/sh
# Run the command as the owner of /data, so outputs written to a mounted directory belong to its
# owner, whatever their uid, and subdirectories writable by that owner are writable here.
# A root-owned /data (an unmounted volume, or a root-owned directory) falls back to the kneemesh user.
# Started as another user (docker run --user), the command runs as that user unchanged.
set -e
if [ "$(id -u)" = 0 ]; then
    owner=$(stat -c %u /data)
    group=$(stat -c %g /data)
    if [ "$owner" = 0 ]; then
        owner=$(id -u kneemesh)
        group=$(id -g kneemesh)
    fi
    exec setpriv --reuid="$owner" --regid="$group" --clear-groups -- "$@"
fi
exec "$@"
