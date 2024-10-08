# Generated by Django 4.2.6 on 2024-02-11 17:30

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mindnet', '0038_alter_pipejob_epoch_from'),
    ]

    operations = [
        migrations.AlterField(
            model_name='pipejob',
            name='duration',
            field=models.FloatField(blank=True, default=4.0),
        ),
        migrations.AlterField(
            model_name='pipejob',
            name='overlap',
            field=models.FloatField(blank=True, default=1.0),
        ),
    ]
