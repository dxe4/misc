-module(drop3).
-export([fall_velocity/2]).

fall_velocity(Planemo, Distance) ->
  Gravity = case Planemo of
              earth -> 9.8;
              moon -> 1.6;
              mars -> 3.71
            end,
  try math:sqrt(2 * Gravity * Distance)
  catch
    error:Error -> {error, Error}
  end.


%% try math:sqrt(2 * Gravity * Distance) of
%%  Result -> Result
%% catch
%%  error:Error -> {error, Error}
%% end.

%%
%% try some:function(argument)
%% catch
%%  error:Error -> {found, Error};
%%  throw:Exception -> {caught, Exception}
%% end;