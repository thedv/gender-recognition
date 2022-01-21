# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from datetime import datetime
from django.db import models

# Create your models here.
class Records(models.Model):
    id = models.AutoField(primary_key=True)
    gender = models.CharField(max_length=50)
    ageGroup = models.CharField(max_length=50, null=True)
    isActive = models.BooleanField(default = True)
    def __str__(self):
        return str(self.id)
    class Meta:
        verbose_name_plural = "ids"
