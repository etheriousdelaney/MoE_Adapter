#!/usr/bin/env bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.
#           2014  David Snyder

# This script combines the data from multiple source directories into
# a single destination directory.

# See http://kaldi-asr.org/doc/data_prep.html#data_prep_data for information
# about what these directories contain.

# Begin configuration section.
extra_files= # specify additional files in 'src-data-dir' to merge, ex. "file1 file2 ..."
skip_fix=false # skip the fix_data_dir.sh in the end
# End configuration section.

echo "$0 $@"  # Print the command line for logging

# if [ -f path.sh ]; then . ./path.sh; fi
# . parse_options.sh || exit 1;

if [ $# -lt 2 ]; then
  echo "Usage: combine_data.sh [--extra-files 'file1 file2'] <dest-data-dir> <src-data-dir1> <src-data-dir2> ..."
  echo "Note, files that don't appear in all source dirs will not be combined,"
  echo "with the exception of utt2uniq and segments, which are created where necessary."
  exit 1
fi

dest=$1;
shift;

first_src=$1;

rm -r $dest 2>/dev/null || true
mkdir -p $dest;

export LC_ALL=C

for dir in $*; do
  if [ ! -f $dir/utt2spk ]; then
    echo "$0: no such file $dir/utt2spk"
    exit 1;
  fi
done



for file in utt2spk utt2lang utt2dur utt2num_frames reco2dur feats.scp text cmvn.scp vad.scp reco2file_and_channel wav.scp spk2gender $extra_files; do
  exists_somewhere=false
  absent_somewhere=false
  for d in $*; do
    if [ -f $d/$file ]; then
      exists_somewhere=true
    else
      absent_somewhere=true
      fi
  done

  if ! $absent_somewhere; then
    set -o pipefail
    ( for f in $*; do cat $f/$file; done ) | sort -k1 > $dest/$file || exit 1;
    set +o pipefail
    echo "$0: combined $file"
  else
    if ! $exists_somewhere; then
      echo "$0 [info]: not combining $file as it does not exist"
    else
      echo "$0 [info]: **not combining $file as it does not exist everywhere**"
    fi
  fi
done

./utils/utt2spk_to_spk2utt.pl <$dest/utt2spk >$dest/spk2utt

if [[ $dir_with_frame_shift ]]; then
  cp $dir_with_frame_shift/frame_shift $dest
fi

if ! $skip_fix ; then
  ./utils/fix_data_dir.sh $dest || exit 1;
fi

exit 0
