import os

def split_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract first style block
    style_start = content.find('<style>')
    style_end = content.find('</style>')
    
    css_content = ""
    if style_start != -1 and style_end != -1:
        css_content = content[style_start+7:style_end].strip()
        content = content[:style_start] + '<link rel="stylesheet" href="/static/styles.css">' + content[style_end+8:]

    # Extract script block with actual logic (not the CDN links)
    script_start = content.find('<script>\n')
    if script_start == -1:
        script_start = content.find('<script>')
        
    script_end = content.rfind('</script>')
    
    js_content = ""
    if script_start != -1 and script_end != -1:
        js_content = content[script_start+8:script_end].strip()
        content = content[:script_start] + '<script src="/static/app.js"></script>' + content[script_end+9:]

    # Write CSS
    with open(os.path.join(os.path.dirname(file_path), 'styles.css'), 'w', encoding='utf-8') as f:
        f.write(css_content)

    # Write JS
    with open(os.path.join(os.path.dirname(file_path), 'app.js'), 'w', encoding='utf-8') as f:
        f.write(js_content)

    # Write HTML
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    split_html('src/static/index.html')
