CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -fPIC -std=c11 -I.
LDFLAGS = -shared -lm

TARGET  = libtopo_engine.so
SRCS    = topo_node.c
OBJS    = $(SRCS:.c=.o)

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^

%.o: %.c topo_node.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

test: $(TARGET)
	@echo "Library built successfully: $(TARGET)"
	@ls -lh $(TARGET)
