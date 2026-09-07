# PyFlink needs a Python environment matching apache-flink's own bundled
# protocol version, plus the Kafka connector on the JVM classpath — neither
# of which the plain flink:1.18 image ships with.
FROM flink:1.18.1-scala_2.12-java11

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip python3-dev && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir apache-flink==1.18.1 redis

# The DataStream Kafka connector — not bundled with Flink itself.
ADD https://repo1.maven.org/maven2/org/apache/flink/flink-sql-connector-kafka/3.1.0-1.18/flink-sql-connector-kafka-3.1.0-1.18.jar \
    /opt/flink/lib/flink-sql-connector-kafka-3.1.0-1.18.jar

COPY src/streaming/flink_job.py /opt/flink-job/flink_job.py
