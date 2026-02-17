# Jazz-Hands
E-SONIC glove-based musical instrument
___
# Box Initialization plans
* Upon first button press, set that position as the origin
```


                     x
            
            
```
* Upon button release, set that as the boundary of the box
```
          
         _ _ _ _ _ _ _ _ _ _ _ _ _
         |                       |
         |                       |
         |                       |
         |            x--------> |
         |                       |
         |                       |
         |                       |
         - - - - - - - - - - - - - 
         
```
* Therefore, we draw the box from the centre to the midpoint of the side of the square



# Main Workflow:
## 1. Get a packet

## 2. Update glove values

## 3. Do logic with position to find pitch using simple ratios

```
          
          _ _ _ _ _ _ _ _ _ _ _ _ _
          |      |                |
          |      |     20% x      |
          |------x----------------|
          |      |                |
          |      | 65% y          |
          |      |                |
          |      |                |
          - - - - - - - - - - - - - 
          
```

## 4. send to VisPy

## 5. send to MIDI
# Important Info!
## Arduino IDE setup guide
### Board Setting
The esp32 board library has examples for ESP_NOW (both broadcast master and slave)
* Go to boards manager and download `esp32` by `Expressif Systems`
  * Set your board to `ESP 32 Dev Module` (IMPORTANT, found out the hard way that other esp32 modules won't work the same)
### Component Libraries
All the libraries have great examples!
#### BNO085
Wiring can be found in this pdf: [Adafruit BNO085 User Guide](https://cdn-learn.adafruit.com/downloads/pdf/adafruit-9-dof-orientation-imu-fusion-breakout-bno085.pdf)
* Download `Adafruit BNO08x` by `Adafruit`
* I think all you need to include are `<Wire.h>` and `<Adafruit_BNO08x.h>`
#### UWB (DW1000)
No Wiring because we got an integrated UWB + ESP32n board (yay!)
* Library is a .zip library found here: [Makerfabs GitHub]([https://github.com/Makerfabs/Makerfabs-ESP32-UWB/blob/main/mf_DW1000.zip)
* Download `mf_DW10000.zip` then include it in Arduino IDE using `Sketch > Include Library > Add .ZIP Library`
## ESP 32 Info
### Mac Addresses
| # | ESP32 Serial Address (MacOS)  | ESP-NOW Mac Address | ESP32 Serial Address (Win) | 
|---|-------------------------------|---------------------|----------------------------|
| 1 | /dev/cu.usbserial-023B6AB4    | 34:98:7A:74:39:00   | yall figure this one out   |
| 2 | /dev/cu.usbserial-023B6B01    | 34:98:7A:73:75:B8   | cuz im not gonna use       |
| 3 | /dev/cu.usbserial-023B6AC7    | 34:98:7A:73:93:14   | Win 11 for a while         |
| 4 | /dev/cu.usbserial-023B6B29    | 08:F9:E0:92:C0:08   |                            |


