# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/akshay-sv/esp/esp-idf/components/bootloader/subproject"
  "/home/akshay-sv/live-asl-translation/embedded/esp32_tft/build/bootloader"
  "/home/akshay-sv/live-asl-translation/embedded/esp32_tft/build/bootloader-prefix"
  "/home/akshay-sv/live-asl-translation/embedded/esp32_tft/build/bootloader-prefix/tmp"
  "/home/akshay-sv/live-asl-translation/embedded/esp32_tft/build/bootloader-prefix/src/bootloader-stamp"
  "/home/akshay-sv/live-asl-translation/embedded/esp32_tft/build/bootloader-prefix/src"
  "/home/akshay-sv/live-asl-translation/embedded/esp32_tft/build/bootloader-prefix/src/bootloader-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/akshay-sv/live-asl-translation/embedded/esp32_tft/build/bootloader-prefix/src/bootloader-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/akshay-sv/live-asl-translation/embedded/esp32_tft/build/bootloader-prefix/src/bootloader-stamp${cfgdir}") # cfgdir has leading slash
endif()
