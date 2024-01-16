from django.shortcuts import render

from cpanel import models as cm
from mindnet import models as mnm


def Scheduled():
    print("Hello I'm Scheduled function")

def Scheduled_hook(f):
    print("Hello I'm Scheduled Hook function")