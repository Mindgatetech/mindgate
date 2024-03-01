import sys, os, django
sys.path.append("config") #here store is root folder(means parent).
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()
from kafka import KafkaConsumer
from django.conf import settings
from mindnet import models
def consumer_create(topic, bootstrap_servers):
    c = KafkaConsumer(
        topic, bootstrap_servers=bootstrap_servers,
        group_id='mindstream_group',
        auto_offset_reset='latest')
    return c
bootstrap_servers = settings.KAFKA_SERVER
consumer_root = consumer_create('root_token', bootstrap_servers=bootstrap_servers)
print("consumer_roo is ok")
while True:
    for token in consumer_root.topics():
        consumer = consumer_create(token, bootstrap_servers=bootstrap_servers)
        print(consumer)
        nameList = list()
        for message in consumer:
            print("%s:%d:%d: key=%s" % (message.topic, message.partition,
                                        message.offset, message.key,
                                        ))
            name = '_' + message.key.decode('UTF-8')
            print(name)
            if 'ENDofMESSAGE' not in name:
                with open(name, 'a+b') as f:
                    f.write(message.value)
                nameList.append(name)
                auxiliaryList = list(set(nameList))
                nameList.pop()
                nameList = auxiliaryList.copy()
            else:
                print(nameList)
        consumer.close()

