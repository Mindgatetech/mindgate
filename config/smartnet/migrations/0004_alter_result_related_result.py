# Generated by Django 4.2.6 on 2024-01-22 15:25

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('smartnet', '0003_result_related_result'),
    ]

    operations = [
        migrations.AlterField(
            model_name='result',
            name='related_result',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='smartnet.result'),
        ),
    ]
