{-# LANGUAGE BangPatterns #-}

-- HaskellBrot: persistent Mandelbrot worker.
-- Protocol: stdin line "<w> <h> <iters>\n" -> stdout w*h*2 bytes uint16 LE row-major.

module Main (main) where

import Control.Concurrent (forkOn, newEmptyMVar, putMVar, takeMVar)
import Control.Monad (forM, forM_, unless, when)
import qualified Data.ByteString as B
import Data.ByteString.Internal (fromForeignPtr)
import Data.IORef (IORef, atomicModifyIORef', newIORef)
import Data.Word (Word16)
import Foreign.ForeignPtr (ForeignPtr, castForeignPtr, mallocForeignPtrArray, withForeignPtr)
import Foreign.Marshal.Utils (copyBytes)
import Foreign.Ptr (Ptr, plusPtr)
import Foreign.Storable (pokeElemOff)
import GHC.Conc (numCapabilities)
import System.IO
import Text.Read (readMaybe)

main :: IO ()
main = do
  hSetBinaryMode stdout True
  hSetBuffering stdout (BlockBuffering Nothing)
  hSetBuffering stdin LineBuffering
  serve

serve :: IO ()
serve = do
  eof <- isEOF
  unless eof $ do
    line <- hGetLine stdin
    case mapM readMaybe (words line) of
      Just [w, h, it] | w > 0 && h > 0 -> do
        frame <- render w h it
        B.hPut stdout frame
        hFlush stdout
      _ -> return ()
    serve

-- Escape-time count (iterations survived; in-set = maxIter).
pixel :: Double -> Double -> Int -> Word16
pixel !cr !ci !maxIter
  | q * (q + crm) <= 0.25 * ci2 = m -- cardioid
  | cp1 * cp1 + ci2 <= 0.0625 = m   -- period-2 bulb
  | otherwise = go 0.0 0.0 0
  where
    !m = fromIntegral maxIter
    !crm = cr - 0.25
    !ci2 = ci * ci
    !q = crm * crm + ci2
    !cp1 = cr + 1.0
    go :: Double -> Double -> Int -> Word16
    go !zr !zi !it
      | it >= maxIter = m
      | zr2 + zi2 > 4.0 = fromIntegral it
      | otherwise = go (zr2 - zi2 + cr) (2.0 * zr * zi + ci) (it + 1)
      where
        !zr2 = zr * zr
        !zi2 = zi * zi
{-# INLINE pixel #-}

renderRow :: Ptr Word16 -> Int -> Int -> Int -> Double -> Double -> Int -> IO ()
renderRow !ptr !w !h !maxIter !dx !dy !row = do
  let !ci = -1.0 + fromIntegral row * dy
      !off = row * w
      go !x
        | x >= w = return ()
        | otherwise = do
            let !cr = -2.5 + fromIntegral x * dx
            pokeElemOff ptr (off + x) (pixel cr ci maxIter)
            go (x + 1)
  go 0
  -- y-axis symmetry: mirror the row (no-op guard for the odd middle row)
  let !mrow = h - 1 - row
  when (mrow /= row) $
    copyBytes (ptr `plusPtr` (mrow * w * 2)) (ptr `plusPtr` (off * 2)) (w * 2)

render :: Int -> Int -> Int -> IO B.ByteString
render !w !h !maxIter = do
  fp <- mallocForeignPtrArray (w * h) :: IO (ForeignPtr Word16)
  withForeignPtr fp $ \ptr -> do
    let !halfRows = (h + 1) `div` 2
        !dx = 3.5 / fromIntegral (max 1 (w - 1))
        !dy = 2.0 / fromIntegral (max 1 (h - 1))
    next <- newIORef 0 :: IO (IORef Int)
    let worker = do
          row <- atomicModifyIORef' next (\n -> (n + 1, n))
          when (row < halfRows) $ do
            renderRow ptr w h maxIter dx dy row
            worker
    dones <- forM [0 .. numCapabilities - 1] $ \i -> do
      mv <- newEmptyMVar
      _ <- forkOn i (worker >> putMVar mv ())
      return mv
    forM_ dones takeMVar
  return $ fromForeignPtr (castForeignPtr fp) 0 (w * h * 2)
