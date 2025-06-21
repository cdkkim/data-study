FROM quay.io/astronomer/astro-runtime:12.4.0

USER root

# Install OpenJDK-17
RUN apt update && \
    apt-get install -y openjdk-17-jdk && \
    apt-get install -y ant && \
    apt-get clean;

# Set JAVA_HOME
ENV JAVA_HOME /usr/lib/jvm/java-17-openjdk-arm64
RUN export JAVA_HOME

USER astroROM astrocrpublic.azurecr.io/runtime:3.0-4
