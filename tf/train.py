#!/usr/bin/python3

import os
import struct
import chess
import numpy as np
import keras
from keras.models import Model, Sequential
from keras import layers
import tensorflow as tf;
from tqdm import tqdm;

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.keras.utils.disable_interactive_logging()


import csv

data_path = os.path.join(os.path.dirname(__file__), '../data/fivem.csv') 
RESIDUAL_BLOCKS =16

noise_level = 64

l2reg = keras.regularizers.l2(l=0.5 * (0.0001))

def conv_2d(generator, filters):

   generator.add(layers.Conv2D(filters, 
                               kernel_size=3, 
                               use_bias=False, 
                               padding="same",
                               kernel_initializer='glorot_normal',
                               kernel_regularizer=l2reg
                               ))
   generator.add(layers.LeakyReLU(0.2))
   generator.add(layers.Dropout(0.3))



def build_generator():

    generator = Sequential()

    generator.add(layers.Dense(units=256, input_dim=noise_level))


    generator.add(layers.LeakyReLU(0.2))

    generator.add(layers.Dense(units=512))

    generator.add(layers.Reshape([8, 8, 8]))
    for i in range(RESIDUAL_BLOCKS):
      conv_2d(generator, 16)

    generator.add(layers.LeakyReLU(0.2))

    generator.add(layers.Flatten())
    generator.add(layers.Dense(units=13*8*8, activation='sigmoid'))

    generator.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))

    return generator


def build_discriminator():

    discriminator = Sequential()

    discriminator.add(layers.Dense(units=1024, input_dim=13*8*8))
    discriminator.add(layers.LeakyReLU(0.2))
    discriminator.add(layers.Dropout(0.2))

    discriminator.add(layers.Reshape([16, 8, 8]))
    for i in range(RESIDUAL_BLOCKS):
      conv_2d(discriminator, 8)

    discriminator.add(layers.Flatten())
    discriminator.add(layers.Dense(units=1, activation='sigmoid'))

    discriminator.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))

    return discriminator



def gan_net(generator, discriminator):

    discriminator.trainable = False

    inp=layers.Input(shape=noise_level)

    X=generator(inp)
    out=discriminator(X)

    gan=Model(inputs=inp, outputs=out)

    gan.compile(loss='binary_crossentropy', optimizer='adam')

    return gan


def train(X_train, epochs, batch_size):

    generator=build_generator()
    discriminator=build_discriminator()
    gan=gan_net(generator, discriminator)

    for epoch in tqdm(range(1, epochs + 1)):
        print("## @ Epoch ", epoch)
        
        for _ in range(batch_size):
            noise = gen_fen_noise(batch_size)

            generated_images = generator.predict(noise)
            image_batch = X_train[np.random.randint(low=0, high=X_train.shape[0], size=batch_size)]
            X=np.concatenate([image_batch, generated_images])

            y_dis=np.zeros(2*batch_size)
            y_dis[:batch_size]=1.0

            discriminator.trainable=True
            discriminator.train_on_batch(X,y_dis)

            noise=gen_fen_noise(batch_size)
            y_gen=np.ones(batch_size)

            discriminator.trainable=False

            gan.train_on_batch(noise,y_gen)


            if epoch==1 or epoch %4 == 0:
                plot_fen(epoch, generator)


def gen_fen_noise(batch_size):
    return np.random.normal(0, 1, [batch_size, noise_level])



def plot_fen(epoch, generator):
    noise=gen_fen_noise(1)

    generated_position=generator.predict(noise)

    generated_position = generated_position.reshape(-1, 13, 8, 8)

    generated_position = generated_position[0]

    unpack_print(generated_position)

def unpack_print(generated_position):
    board = chess.Board.empty()

    generated_position = generated_position.reshape(-1, 64)
    generated_position = np.where(generated_position > 0.5, 1, 0)

    whites = chess.SquareSet(int.from_bytes(np.packbits(generated_position[6]), 'little'))
    blacks = chess.SquareSet(int.from_bytes(np.packbits(generated_position[7]), 'little'))

    for sq in whites & chess.SquareSet(int.from_bytes(np.packbits(generated_position[0]), 'little')):
        board.set_piece_at(sq, chess.Piece(chess.ROOK, chess.WHITE))
    for sq in whites & chess.SquareSet(int.from_bytes(np.packbits(generated_position[1]), 'little')):
        board.set_piece_at(sq, chess.Piece(chess.KNIGHT, chess.WHITE))
    for sq in whites & chess.SquareSet(int.from_bytes(np.packbits(generated_position[2]), 'little')):
        board.set_piece_at(sq, chess.Piece(chess.BISHOP, chess.WHITE))
    for sq in whites & chess.SquareSet(int.from_bytes(np.packbits(generated_position[3]), 'little')):
        board.set_piece_at(sq, chess.Piece(chess.QUEEN, chess.WHITE))
    for sq in whites & chess.SquareSet(int.from_bytes(np.packbits(generated_position[4]), 'little')):
        board.set_piece_at(sq, chess.Piece(chess.KING, chess.WHITE))
    for sq in whites & chess.SquareSet(int.from_bytes(np.packbits(generated_position[5]), 'little')):
        board.set_piece_at(sq, chess.Piece(chess.PAWN, chess.WHITE))
 
    for sq in blacks & chess.SquareSet(int.from_bytes(np.packbits(generated_position[0]), 'little')):
        board.set_piece_at(sq, chess.Piece(chess.ROOK, chess.BLACK))
    for sq in blacks & chess.SquareSet(int.from_bytes(np.packbits(generated_position[1]), 'little')):
        board.set_piece_at(sq, chess.Piece(chess.KNIGHT, chess.BLACK))
    for sq in blacks & chess.SquareSet(int.from_bytes(np.packbits(generated_position[2]), 'little')):
        board.set_piece_at(sq, chess.Piece(chess.BISHOP, chess.BLACK))
    for sq in blacks & chess.SquareSet(int.from_bytes(np.packbits(generated_position[3]), 'little')):
        board.set_piece_at(sq, chess.Piece(chess.QUEEN, chess.BLACK))
    for sq in blacks & chess.SquareSet(int.from_bytes(np.packbits(generated_position[4]), 'little')):
        board.set_piece_at(sq, chess.Piece(chess.KING, chess.BLACK))
    for sq in blacks & chess.SquareSet(int.from_bytes(np.packbits(generated_position[5]), 'little')):
        board.set_piece_at(sq, chess.Piece(chess.PAWN, chess.BLACK))
       
    print(board)

pieces = [
    chess.ROOK,
    chess.KNIGHT,
    chess.BISHOP,
    chess.QUEEN,
    chess.KING,
    chess.PAWN
]
"""r n b q k p b w""" """ 8 x 8 + 5 * 8 = 13 * 8"""
"""rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"""
def pack_fen_str(fen):
    x = bytearray()
    board = chess.Board(fen)
    whites = chess.SquareSet()
    blacks = chess.SquareSet()
    for p in pieces:
      pp = int(board.pieces(p, chess.WHITE) | board.pieces(p, chess.BLACK))
      x.extend(pp.to_bytes(8, 'little'))
      whites |= board.pieces(p, chess.WHITE)
      blacks |= board.pieces(p, chess.BLACK)
    

    x.extend(int(whites).to_bytes(8, 'little'))
    x.extend(int(blacks).to_bytes(8, 'little'))

    x.extend(int(0).to_bytes(8, 'little'))
    x.extend(int(0).to_bytes(8, 'little'))
    x.extend(int(0).to_bytes(8, 'little'))
    x.extend(int(0).to_bytes(8, 'little'))
    x.extend(int(0).to_bytes(8, 'little'))

    return x

struct_string = '104s'

def get_x_train():

    #file = open('athousand_sorted.csv')
    file = open(data_path)
    csvreader = csv.reader(file)
    fens = []
    for row in csvreader:
        fen = row[1]
        moves = row[2]
        move = moves.split(' ')[0]
        board = chess.Board(fen)
        board.push(chess.Move.from_uci(move))
        fens.append(pack_fen_str(board.fen()))

    stuff = bytearray()
    for content in fens:
      planes = struct.Struct(struct_string).unpack(content)[0]
  
      planes = np.unpackbits(np.frombuffer(planes, dtype=np.uint8))

      stuff += bytearray(planes)

    stuff = np.array(stuff).reshape(-1, 13*8*8)
  
    return stuff
    
    
def test_pack():

    ls = np.array(pack_fen_str("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"))
    ls = ls.reshape(-1, 13, 8)
    unpack_print(ls[0])


if __name__ == "__main__":
    X_train=get_x_train()
    train(X_train, epochs=500, batch_size = 128)

    #test_pack()

