with open('amazon_api_resources.txt', 'r') as f:
    output = set()
    for line in f.readlines():
        line = line.strip()
        for item in line.split(','):
            item = item.strip()
            if item in output:
                pass
            else:
                output.add(item)
    print(output)
with open('amazon_api_concepts.txt', 'w+') as f:
    for item in output:
        f.writelines(item+'\n')