# Generated by Django 4.2.6 on 2024-01-15 17:28

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('mindnet', '0013_pipejob_scaler'),
    ]

    operations = [
        migrations.RenameField(
            model_name='pipejob',
            old_name='Stim_Chan',
            new_name='stim_channel',
        ),
    ]
