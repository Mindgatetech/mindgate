# Generated by Django 4.2.6 on 2024-02-14 14:49

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dev', '0006_alter_token_secret'),
    ]

    operations = [
        migrations.AlterField(
            model_name='token',
            name='secret',
            field=models.CharField(blank=True, max_length=16, unique=True),
        ),
    ]
