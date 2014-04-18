-module(bounce).
-export([report/1]).

report(Count) ->
  receive
    X -> io:format("Received #~p: ~p~n",[Count,X]),
    report(Count+1)
  end.

%% 1> c(bounce).
%% {ok,bounce}
%% 2> Pid=spawn(bounce,report,[]).
%% <0.41.0>
%% 3> Pid ! 23.
%% Received 23
%% 23
%% 4>

% register
% Pid1=spawn(bounce,report,[1]).
% register(bounce,Pid1).
% bounce ! hello.