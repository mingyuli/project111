import os
import sys
import csv

import Agent
from ProblemSet import ProblemSet
from RavensGrader import grade

def getNextLine(r):
    return r.readline().rstrip()

def solve():
    sets=[]
    r = open(os.path.join("Problems","ProblemSetList.txt"))
    line = getNextLine(r)
    while not line=="":
        sets.append(ProblemSet(line))
        line=getNextLine(r)
    agent=Agent.Agent()
    with open("AgentAnswers.csv","w") as results:
        results.write("ProblemSet,RavensProblem,Agent's Answer\n")
        for set in sets:
            for problem in set.problems:
                answer = agent.Solve(problem)
                results.write("%s,%s,%d\n" % (set.name, problem.name, answer))
    r.close()

def main():
    solve()
    grade()

if __name__ == "__main__":
    main()