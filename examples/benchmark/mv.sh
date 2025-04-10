
cd ./refit_data
find ./ -type f -name 'refit_*' -exec sh -c '
  for f do
    mv "$f" "${f/refit_/full_}"
  done
' sh {} +