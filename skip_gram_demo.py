from collections import Counter


def run():
    with open("text8.txt", "r") as file:
        text = file.read()
    text_list = text.strip().lower().split(" ")
    text_count = dict(Counter(text_list))
    text_count = sorted(text_count.items(), key=lambda x: x[1], reverse=True)
    text_count = dict(text_count)
    for index, text in enumerate(text_count.items()):
        text_count[text[0]] = index

    text_list = [text_count[each_text] for each_text in text_list]
    print()


if __name__ == '__main__':
    run()