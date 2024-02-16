import secrets
from django.conf import settings
from kafka.admin import KafkaAdminClient, NewTopic
from kafka import KafkaConsumer

from django.db import models
from cpanel.models import User
from secrets import token_hex
from django.dispatch import receiver
from django.db.models.signals import post_save, post_delete
class Token(models.Model):
    user        = models.ForeignKey(User, on_delete=models.CASCADE, null=True)
    name        = models.CharField(max_length=100, blank=False)
    description = models.TextField(blank=False)
    SUB_OPT     = [('Data Gathering', 'Data Gathering'), ('Online Evaluation', 'Online Evaluation')]
    subject     = models.CharField(max_length=20, choices=SUB_OPT, blank=False)
    secret      = models.CharField(max_length=16, unique=True, blank=True)
    STATUS_OPT = [('Permissioned', 'Permissioned'), ('Rejected', 'Rejected'), ('Pending', 'Pending')]
    status = models.CharField(max_length=15, choices=STATUS_OPT, default='Pending', blank=False)

    def __str__(self):
        return self.name


@receiver(post_save, sender=Token)
def token_generator(sender, instance, created, **kwargs):
    if created:
        secret = secrets.token_hex(8)
        print(secret)
        instance.secret = secret
        instance.save()

@receiver(post_save, sender=Token)
def token_register(sender, instance, created, **kwargs):
    if not created:
        print('token_register passed...')
        kafka_admin_client = KafkaAdminClient(bootstrap_servers=settings.KAFKA_SERVER)
        consumer = KafkaConsumer(
            'root_token',
            bootstrap_servers=settings.KAFKA_SERVER,
            group_id='mindstream_group',)
        topics = consumer.topics()
        if instance.status == 'Permissioned':
            print('Permissioned passed...')
            if instance.secret not in topics:
                print('instance.secret passed...')
                topic_list = [(NewTopic(name=instance.secret, num_partitions=1, replication_factor=1)),]
                kafka_admin_client.create_topics(new_topics=topic_list)
        else:
            print('Rejected and Pending passed...')
            if instance.secret in topics:
                print('deletion passed...')
                kafka_admin_client.delete_topics(topics=[instance.secret, ])

@receiver(post_delete, sender=Token)
def token_unregister(sender, instance, **kwargs):
    kafka_admin_client = KafkaAdminClient(bootstrap_servers=settings.KAFKA_SERVER)
    consumer = KafkaConsumer(
        'root_token',
        bootstrap_servers=settings.KAFKA_SERVER,
        group_id='mindstream_group', )
    topics = consumer.topics()
    if instance.secret in topics:
        print('deletion passed...')
        kafka_admin_client.delete_topics(topics=[instance.secret, ])