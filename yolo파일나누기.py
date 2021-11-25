import glob


def file_path_save():
    filenames = []

    files = sorted(glob.glob("data/test_img/*.jpg"))

    for i in range(len(files)):
        f = open("test.txt", 'a')
        f.write(files[i] + "\n")


if __name__ == '__main__':
    file_path_save()
