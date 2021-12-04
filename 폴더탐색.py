import os

def print_files_in_dir(root_dir, prefix):
    files = os.listdir(root_dir)
    for file in files:
        path = os.path.join(root_dir, file)
        print(prefix + path)


if __name__ == "__main__":
    root_dir = "./test/"
    print_files_in_dir(root_dir, "")
    

# os.listdir() 재귀
import os

def print_files_in_dir(root_dir, prefix):
    files = os.listdir(root_dir)
    for file in files:
        path = os.path.join(root_dir, file)
        print(prefix + path)
        if os.path.isdir(path):
            print_files_in_dir(path, prefix + "    ")

if __name__ == "__main__":
    root_dir = "./test/"
    print_files_in_dir(root_dir, "")
    
# os.walk()를 이용한 방법
import os

if __name__ == "__main__":
    root_dir = "./test/"
    for (root, dirs, files) in os.walk(root_dir):
        print("# root : " + root)
        if len(dirs) > 0:
            for dir_name in dirs:
                print("dir: " + dir_name)

        if len(files) > 0:
            for file_name in files:
                print("file: " + file_name)
