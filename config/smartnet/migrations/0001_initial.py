# Generated by Django 4.2.6 on 2024-01-21 07:16

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Result',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('result_id', models.CharField(max_length=32)),
                ('result', models.TextField(blank=True)),
            ],
        ),
    ]
