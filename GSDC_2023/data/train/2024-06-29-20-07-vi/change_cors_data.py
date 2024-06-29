import re

# Read the content of the file
with open('hnoi1810.24o', 'r') as file:
    content = file.read()

# Define a regular expression pattern
pattern = r'([GRJCE])\s'

# Replace spaces with '0' if the word before space is G, R, J, C, or E
modified_content = re.sub(pattern, r'\g<1>0', content)

# Write the modified content back to the file
with open('hnoi1810.24o', 'w') as file:
    file.write(modified_content)