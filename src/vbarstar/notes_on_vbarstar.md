1. Float velocity (u1,v1, u2,v2) that you received in float files is in GEOMAGNETIC coordinate. The magnetic variation, magvar, (or called declination) is recorded in float's velocity raw data file.

2. Float velocity (u1,v1,u2,v2) are relative to an unknown 'depth constant' velocity', called Vbarstar. Vbarstar can be a function of time, sediment conductivity (topography), etc.

3. Vbarstar can be computed by comparing GPS positions (before
descending and after ascending) derived velocity with the averaged
velocity computed from (u1, v1, u2, v2).

4. Cautions for GPS derived velocity:
          a. Use the right GPS clock time (there are several variables of time stamp in GPS file and they are slightly different). Check my vstar_fun.m for the right time.
          b. There is always a time gap between the GPS fix and EM-APEX float velocity. Float cannot get GPS when it is below surface. Also, it does not start sampling velocity immediately below the surface. It is crucial to take into account of this time gap. The best way is to extrapolate GPS positions forward in time to the time of the first velocity measurements. Similarly, for the ascending part of velocity and GPS position. GPS positions should be backward in time to the last float velocity measurements.
          c. The above extrapolation is done linearly.
          d. Difference between GPS positions divided by the time
interval yields the averaged GPS velocity. This should be the real
ABSOLUTE average float velocity. THIS IS IN Geographical coordinate.
          e. Call them Ugps, Vgps

5. Cautions for EM-APEX float velocity:
          a. These velocity (u1,v1,u2,v2) are in Geomagnetic coordinate.
          b. Use only the good quality data justified by verr1, verr2,
RotP, W
          c. DON'T simply averaging the velocity (u1, v1, u2, v2),
because as the float turns around at depths, it does not take velocity
measurements, often O(10) minutes. This missing data could cause
              large error of Vbarstar if not taken into account carefully.
          d. Instead, integrating (u1, v1, u2, v2) with time, linearly
interpolate over NaN, yields the total distance dX1, dY1, dX2, dY2.
Dividing them by the time difference yields dX1/dt, dY1/dt, dX2/dt, dY2/dt.
              Call these U1, V1, U2, V2.
          e. Rotate them to Geographical coordinate using magvar, Uf1,
Vf1, Uf2, Vf2.
6. Ubarstar1 = Ugps - Uf1, Vbarstar2 = Vgps - Vf1.
          Ubarstar2 = Ugps - UF2, Vbarstar2 = Vgps - Vf2

7. Extra caution when floats are yoyoing at depths for a long period of
missing GPS positions. I have an modified vstar_yoyo_fun.m to deal with it.
8. For floats park at depth for a long long period of time without
velocity measurements (floats need to profile to get velocity
measurements), it would be challenging to compute Vbarstar.
