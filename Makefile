CC = gcc
CFLAGS = -Wall -Wextra -O2

SRC_DIR = src
INC_DIR = include

SRC_FILES = $(SRC_DIR)/main.c $(SRC_DIR)/mlp.c $(SRC_DIR)/layer.c $(SRC_DIR)/neuron.c $(SRC_DIR)/dataset.c 
OBJ_FILES = $(SRC_FILES:.c=.o)

TARGET = mlp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJ_FILES)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJ_FILES)

%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -I$(INC_DIR) -c $< -o $@

clean:
	rm -f $(OBJ_FILES) $(TARGET)
