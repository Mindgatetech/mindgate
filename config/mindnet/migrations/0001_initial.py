# Generated by Django 4.2.6 on 2024-01-15 09:39

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Paper',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(default=None, max_length=200)),
                ('abstract', models.TextField(default=None)),
                ('published_year', models.IntegerField(default=None)),
                ('doi', models.CharField(default=None, max_length=300)),
                ('private', models.BooleanField(default=True)),
                ('user', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, related_name='user_paper', to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='Dataset',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(default=None, max_length=25)),
                ('description', models.TextField(blank=True, default=None)),
                ('type', models.CharField(choices=[('EEG', 'EEG'), ('fMRI', 'fMRI')], default='EEG', max_length=10)),
                ('research_field', models.CharField(choices=[('MI', 'Motor Imagery'), ('ERP', 'ERP')], default='MI', max_length=20)),
                ('eeg_channels', models.CharField(blank=True, max_length=256, null=True)),
                ('eog_channels', models.CharField(blank=True, max_length=256, null=True)),
                ('dataset_link', models.URLField(max_length=500, null=True)),
                ('dataset_path', models.CharField(blank=True, max_length=256, null=True)),
                ('private', models.BooleanField(default=False)),
                ('ready_to_use', models.BooleanField(default=False)),
                ('related_paper', models.ManyToManyField(blank=True, default=None, to='mindnet.paper')),
                ('user', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, related_name='user_dataset', to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]
