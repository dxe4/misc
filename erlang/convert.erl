-module(convert).
-export([mps_to_mph/1, mps_to_kph/1]).

%%% doc
mps_to_mph(Mps) when Mps>= 0 -> 2.23693629 * Mps.
mps_to_kph(Mps) -> 3.6 * Mps.