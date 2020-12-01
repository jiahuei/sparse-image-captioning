#!/usr/bin/env bash
# This script copies Flickr8k images from Google Drive

echo "This script assumes Drive is mounted at '/content/drive'."

if [ -d /content/flickr8k/images ]; then
    echo "Found '/content/flickr8k/images'."
else
    echo "Copying..."
    mkdir '/content/flickr8k'
    cp '/content/drive/My Drive/datasets/flickr8k/Flickr8k_Dataset.zip' '/content/flickr8k/Flickr8k_Dataset.zip'
    unzip -qq -d '/content/flickr8k' '/content/flickr8k/Flickr8k_Dataset.zip'
    mv '/content/flickr8k/Flicker8k_Dataset' '/content/flickr8k/images'
    rm -r '/content/flickr8k/__MACOSX'
  echo "Done."
fi
