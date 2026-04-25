"""This file contains a range of basic utilities used throughout the program"""

def int_key_hook(d):
    return {int(k) if k.isdigit() else k: v for k, v in d.items()}