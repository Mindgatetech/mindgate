# Generated by Django 4.2.6 on 2024-01-21 08:29

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mindnet', '0026_aimodel_granted'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='scaler',
            name='scaler_code',
        ),
        migrations.AddField(
            model_name='scaler',
            name='granted',
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name='scaler',
            name='scaler',
            field=models.FileField(blank=True, default=None, upload_to='Scalers/'),
        ),
    ]
