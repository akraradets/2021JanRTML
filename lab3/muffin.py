import requests
def is_downloadable(url):
    """
    Does the url contain a downloadable resource
    """
    try:
        h = requests.head(url, allow_redirects=True)
    except:
        return False
    header = h.headers
    content_type = header.get('content-type')
    if content_type == None:
        return False
    if 'text' in content_type.lower():
        return False
    if 'html' in content_type.lower():
        return False
    return True

file1 = open('muffin.txt', 'r')
Lines = file1.readlines()
count = 0
for line in Lines:
    print(line)
    if(is_downloadable(line)):
        count += 1
        r = requests.get(line, allow_redirects=True)
        open(f'data/muffin/{count}.jpg', 'wb').write(r.content)