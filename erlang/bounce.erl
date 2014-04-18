-module(bounce).
-export([report/0]).
report() ->
  receive
    X -> io:format("Received ~p~n",[X])
  end.

%% 1> c(bounce).
%% {ok,bounce}
%% 2> Pid=spawn(bounce,report,[]).
%% <0.41.0>
%% 3> Pid ! 23.
%% Received 23
%% 23
%% 4>