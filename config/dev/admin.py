from django.contrib import admin, messages
from django.conf import settings
from . import models

from kafka.admin import KafkaAdminClient, NewTopic
from kafka import KafkaConsumer

class TokenAdmin(admin.ModelAdmin):
    list_display = ('name', 'secret', 'status')
    list_editable = ('status',)
    '''actions = ["RegisteToken",]
    @admin.action(description="Register tokens to kafka cluster")
    def RegisteToken(self, request, queryset):
        kafka_admin_client = KafkaAdminClient(bootstrap_servers=settings.KAFKA_SERVER)
        topic_list = list()
        for obj in queryset:
            topic_list.append(NewTopic(name=obj.secret, num_partitions=1, replication_factor=1))
            obj.status = 'Permissioned'
            obj.save()
        kafka_admin_client.create_topics(new_topics=topic_list)
        messages.success(request, "Successfully token registered!")'''
admin.site.register(models.Token, TokenAdmin)