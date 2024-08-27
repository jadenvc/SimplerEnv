## Write a simple code that prints, and saves a textf iel with hellow wrold in it


import os

def main():
    print("Hello World")
    with open("hello.txt", "w") as f:
        f.write("Hello World")
    print("File saved as hello.txt")
    
if __name__ == "__main__":
    main()