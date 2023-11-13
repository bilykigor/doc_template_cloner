python3.8 -m build
twine check dist/*
twine upload --verbose dist/* 
rm -rf dist
rm -rf *.egg-info
rm -rf build