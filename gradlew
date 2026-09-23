#!/usr/bin/env sh
set -eu
if command -v gradle >/dev/null 2>&1; then
  exec gradle "$@"
fi
GRADLE_VERSION=9.6.0
BASE_DIR="${GRADLE_USER_HOME:-$HOME/.gradle}/cardviper-bootstrap"
DIST_DIR="$BASE_DIR/gradle-$GRADLE_VERSION"
ZIP="$BASE_DIR/gradle-$GRADLE_VERSION-bin.zip"
if [ ! -x "$DIST_DIR/bin/gradle" ]; then
  mkdir -p "$BASE_DIR"
  if [ ! -f "$ZIP" ]; then
    if command -v curl >/dev/null 2>&1; then
      curl -fL "https://services.gradle.org/distributions/gradle-$GRADLE_VERSION-bin.zip" -o "$ZIP"
    elif command -v wget >/dev/null 2>&1; then
      wget -O "$ZIP" "https://services.gradle.org/distributions/gradle-$GRADLE_VERSION-bin.zip"
    else
      echo "CardViper bootstrap needs Gradle, curl, or wget." >&2
      exit 1
    fi
  fi
  rm -rf "$DIST_DIR"
  unzip -q "$ZIP" -d "$BASE_DIR"
fi
exec "$DIST_DIR/bin/gradle" "$@"
