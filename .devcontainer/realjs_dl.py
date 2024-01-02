from io import BytesIO
from zipfile import ZipFile

import requests

response = requests.get("https://github.com/hakimel/reveal.js/archive/master.zip")

zip_file = ZipFile(BytesIO(response.content))
zip_file.extractall("reveal.js")
