# Your Agent for solving Raven's Progressive Matrices. You MUST modify this file.
#
# You may also create and submit new files in addition to modifying this file.
#
# Make sure your file retains methods with the signatures:
# def __init__(self)
# def Solve(self,problem)
#
# These methods will be necessary for the project's main method to run.

# Install Pillow and uncomment this line to access image processing.
from PIL import Image, ImageChops
from copy import deepcopy
from Utils import Transformation, Attribute, Conversion
from TransformationFinder import TransformationFinder
import sys

class Agent:
    # The default constructor for your Agent. Make sure to execute any
    # processing necessary before your Agent starts solving problems here.
    #
    # Do not add any variables to this signature; they will not be used by
    # main().
    def __init__(self):
        sys.setrecursionlimit(10000)
        self.problem_number = 0
        self.DEBUG = False
        self.debugLevel = 1

    # The primary method for solving incoming Raven's Progressive Matrices.
    # For each problem, your Agent's Solve() method will be called. At the
    # conclusion of Solve(), your Agent should return an integer representing its
    # answer to the question: "1", "2", "3", "4", "5", or "6". These integers
    # are also the Names of the individual RavensFigures, obtained through
    # RavensFigure.getName() (as Strings).
    #
    # In addition to returning your answer at the end of the method, your Agent
    # may also call problem.checkAnswer(int givenAnswer). The parameter
    # passed to checkAnswer should be your Agent's current guess for the
    # problem; checkAnswer will return the correct answer to the problem. This
    # allows your Agent to check its answer. Note, however, that after your
    # agent has called checkAnswer, it will *not* be able to change its answer.
    # checkAnswer is used to allow your Agent to learn from its incorrect
    # answers; however, your Agent cannot change the answer to a question it
    # has already answered.
    #
    # If your Agent calls checkAnswer during execution of Solve, the answer it
    # returns will be ignored; otherwise, the answer returned at the end of
    # Solve will be taken as your Agent's answer to this problem.
    #
    # Make sure to return your answer *as an integer* at the end of Solve().
    # Returning your answer as a string may cause your program to crash.
    def Solve(self,problem):
        problem_definition = DefinitionProblem()
        answer = -1
        self.problem_number += 1
        answerChoices = []
        print("Solving " + str(self.problem_number))
        # Solve 2x2 problems for Project 1
        if problem_definition.is_twobytwo_problem(problem):
            answer = TwoByTwoSolver(problem).do_basic_analysis()
            # Return answer if first layer analysis succeeds
            if answer != -1:
                return answer
            # Else perform a deeper analysis
            answer = TwoByTwoSolver(problem).do_transformation_analysis()
        elif problem_definition.is_threebythree_problem(problem):
            #Get images
            A = problem.figures['A'].visualFilename
            A = self.ToBinary(A)
            B = problem.figures['B'].visualFilename
            B = self.ToBinary(B)
            C = problem.figures['C'].visualFilename
            C = self.ToBinary(C)
            D = problem.figures['D'].visualFilename
            D = self.ToBinary(D)
            E = problem.figures['E'].visualFilename
            E = self.ToBinary(E)
            F = problem.figures['F'].visualFilename
            F = self.ToBinary(F)
            G = problem.figures['G'].visualFilename
            G = self.ToBinary(G)
            H = problem.figures['H'].visualFilename
            H = self.ToBinary(H)
            for i in range(1, 9):
                answerChoices.append(problem.figures[str(i)].visualFilename)
            #Solution = self.FindWholeTransforms(A,B,C,D,E,F,G,H)
            #if Solution == -1:
            #Find Tx
            Tx_Hor = self.FindTransformation(A,B,C)
            #self.dispTx(Tx_Hor)
            Tx_Ver = self.FindTransformation(A,D,G)
            #self.dispTx(Tx_Ver)
            Tx_Diag = self.FindDiagTransformation(A,E)

            #Ordering of Txs
            BestTxsDiag = self.GetBestTransformations(Tx_Diag)
            BestTxsHor = self.GetBestTransformations(Tx_Hor)

            #Diag vs HorVert
            if self.DiagVsHorVert(BestTxsDiag,BestTxsHor) == 'Diagonal':
                DiagTxSolutionSet = self.CompareAndGetSolution(A, E, BestTxsDiag, answerChoices)
                answer = self.GetBestSolution(DiagTxSolutionSet)
            else:
                HorTxSolutionSet = self.CompareAndGetSolution(G, H, BestTxsHor, answerChoices)
                #Ordering of Txs
                BestTxsVer = self.GetBestTransformations(Tx_Ver)
                VerTxSolutionSet = self.CompareAndGetSolution(C, F, BestTxsVer, answerChoices,HorTxSolutionSet)
                #Analysing solution set
                answer = self.AnalyseSolutionSet(HorTxSolutionSet, VerTxSolutionSet)
        print(" Solution:"+str(answer))
        return answer

    def FindTransformation(self,A,B,C):
        TxManager = TransformationFinder()
        #Tx = [Tx0,[Tx1,Tx2],[Tx3,Tx4]]
        return TxManager.FindTx(A,B,C)

    def FindDiagTransformation(self,A,B):
        TxManager = TransformationFinder()
        #Tx = [Tx1,Tx3]
        return TxManager.FindDiagTx(A,B);

    def DiagVsHorVert(self, diagTxs, horTxs):
        if diagTxs[0][1] >= horTxs[0][1]:
            if diagTxs[0][1] == horTxs[0][1]:
                    return 'HorVert'
            elif diagTxs[0][1]-horTxs[0][1] <0.5:
                return 'HorVert'
            return 'Diagonal'
        else:
            return 'HorVert'

    def GetBestTransformations(self,Txs):
        BestTxType = Transformation.Empty
        BestTxScore = 0
        BestTxDetails = []
        BestTxsList = []

        BestTxType = Txs[0].getBestTransformation();
        BestTxScore = Txs[0].getHighestScore();
        BestTxDetails = Txs[0].getBestTxDetails();
        BestTxsList.append([BestTxType,BestTxScore,BestTxDetails])
        if len(Txs) > 1:
            if isinstance(Txs[1],list):
                t1 = Txs[1][0]
                t2 = Txs[1][1]
                t=t1
                if t1.getBestTransformation()!=t2.getBestTransformation():
                    t=t2
                BestTxType = t.getBestTransformation();
                BestTxScore = t.getHighestScore();
                BestTxDetails = t.getBestTxDetails();
                if BestTxScore > BestTxsList[0][1]:
                    BestTxsList.insert(0,[BestTxType,BestTxScore,BestTxDetails])
                else:
                    BestTxsList.append([BestTxType,BestTxScore,BestTxDetails])
            else:
                t = Txs[1]
                BestTxType = t.getBestTransformation();
                BestTxScore = t.getHighestScore();
                BestTxDetails = t.getBestTxDetails();
                if BestTxScore > BestTxsList[0][1]:
                    BestTxsList.insert(0,[BestTxType,BestTxScore,BestTxDetails])
                else:
                    BestTxsList.append([BestTxType,BestTxScore,BestTxDetails])
        if len(Txs) > 2:
            if isinstance(Txs[2],list):
                t = Txs[2][1]
                BestTxType = t.getBestTransformation();
                BestTxScore = t.getHighestScore();
                BestTxDetails = t.getBestTxDetails();
                if BestTxScore > BestTxsList[0][1]:
                    BestTxsList.insert(0,[BestTxType,BestTxScore,BestTxDetails])
                elif BestTxScore > BestTxsList[1][1]:
                    BestTxsList.insert(1,[BestTxType,BestTxScore,BestTxDetails])
                else:
                    BestTxsList.append([BestTxType,BestTxScore,BestTxDetails])
            else:
                t = Txs[2]
                BestTxType = t.getBestTransformation();
                BestTxScore = t.getHighestScore();
                BestTxDetails = t.getBestTxDetails();
                if BestTxScore > BestTxsList[0][1]:
                    BestTxsList.insert(0,[BestTxType,BestTxScore,BestTxDetails])
                elif BestTxScore > BestTxsList[1][1]:
                    BestTxsList.insert(1,[BestTxType,BestTxScore,BestTxDetails])
                else:
                    BestTxsList.append([BestTxType,BestTxScore,BestTxDetails])
        return BestTxsList

    def CompareAndGetSolution(self,G,H,BestTx,answerChoices,HorSolSet = None):
        choices = []
        choices.append(0)
        for option in answerChoices:
            choices.append(self.ToBinary(option))
        solution = -1
        Tx = TransformationFinder()
        Tx.BlobsA = Tx.GetBlobs(G)
        Tx.BlobsB = Tx.GetBlobs(H)
        ToCheckChoices = [1,2,3,4,5,6,7,8]
        if HorSolSet != None and len(HorSolSet)>0:
            ToCheckChoices = []
            for k in HorSolSet[:]:
                ToCheckChoices.append(k[0])
        solSet = []
        for t in BestTx[:]:
            BestTxType = t[0]
            BestTxScore = t[1]
            BestTxDetails = t[2]
            if solution == -1:
                for i in ToCheckChoices[:]:
                    #print("Checking ans "+str(i))
                    if BestTxType == Transformation.Same:
                        #print("Checking Tx Empty")
                        score = Tx.Same(H,choices[i])
                        #print("Scores:"+str(score)+":"+str(BestTxScore))
                        if score > 97:
                            #if self.AlmostEqual(score,BestTxScore,2):
                            solution = i
                            solSet.append((i,self.GetDeviation(score,BestTxScore)))
                    elif BestTxType == Transformation.ConstantAddition:
                        #print("Checking Const addition")
                        score, GHAddArea, HIAddArea = Tx.ConstantAddition(G,H,choices[i])
                        #print("Scores:"+str(score)+":"+str(BestTxScore))
                        #print("GHAddarea:"+str(GHAddArea)+" org:"+str(BestTxDetails[0]))
                        #print("HIAddarea:"+str(HIAddArea)+" org:"+str(BestTxDetails[1]))
                        if self.AlmostEqual(score,BestTxScore,1):
                            if self.AlmostEqual(HIAddArea,BestTxDetails[1],1):
                                solution = i
                                solSet.append((i,self.GetDeviation(score,BestTxScore)))
                    elif BestTxType == Transformation.ConstantSubtraction:
                        #print("Checking Const sub")
                        score, GHSubArea, HISubArea = Tx.ConstantSubtraction(G,H,choices[i])
                        #print("Scores:"+str(score)+":"+str(BestTxScore))
                        #print("GHSubarea:"+str(GHSubArea)+" org:"+str(BestTxDetails[0]))
                        #print("HISubarea:"+str(HISubArea)+" org:"+str(BestTxDetails[1]))
                        if self.AlmostEqual(score,BestTxScore,1):
                            if self.AlmostEqual(HISubArea,BestTxDetails[1],1):
                                solution = i
                                solSet.append((i,self.GetDeviation(score,BestTxScore)))
                    elif BestTxType == Transformation.Addition:
                        score, z = Tx.Addition(G,H,choices[i])
                        if self.AlmostEqual(score,BestTxScore,1):
                            solution = i
                            solSet.append((i,self.GetDeviation(score,BestTxScore)))
                    elif BestTxType == Transformation.Subtraction:
                        score, z = Tx.Subtraction(G,H,choices[i])
                        if self.AlmostEqual(score,BestTxScore,1):
                            solution = i
                            solSet.append((i,self.GetDeviation(score,BestTxScore)))
                    elif BestTxType == Transformation.AddcumSub:
                        score, z = Tx.AddcumSub(G,H,choices[i])
                        if self.AlmostEqual(score,BestTxScore,5):
                            solution = i
                            solSet.append((i,self.GetDeviation(score,BestTxScore)))
                    elif BestTxType == Transformation.Common:
                        score, z = Tx.Common(G,H,choices[i])
                        if self.AlmostEqual(score,BestTxScore,2):
                            solution = i
                            solSet.append((i,self.GetDeviation(score,BestTxScore)))
                    elif BestTxType == Transformation.Divergence:
                        #print("Checking Divergence")
                        score, GHScore, GIScore = Tx.Divergence(G,H,choices[i])
                        #print(str(BestTxScore)+","+str(BestTxDetails[0])+","+str(BestTxDetails[1]))
                        #print(str(score)+","+str(GHScore)+","+str(GIScore))
                        if self.AlmostEqual(score,BestTxScore,2):
                            solution = i
                            solSet.append((i,self.GetDeviation(score,BestTxScore)))
                    elif BestTxType == Transformation.Convergence:
                        #print("Checking Convergence")
                        score, GHScore, GIScore = Tx.Convergence(G,H,choices[i])
                        #print(str(BestTxScore)+","+str(BestTxDetails[0])+","+str(BestTxDetails[1]))
                        #print(str(score)+","+str(GHScore)+","+str(GIScore))
                        if self.AlmostEqual(score,BestTxScore,2):
                            solution = i
                            solSet.append((i,self.GetDeviation(score,BestTxScore)))
                    elif BestTxType == Transformation.Migration:
                        Tx.BlobsC = Tx.GetBlobs(choices[i])
                        correspGI = Tx.GetBlobCorrespondence(Tx.BlobsA,Tx.BlobsC)
                        GIMetaData = Tx.GetBlobMetaData(correspGI,Tx.BlobsA,Tx.BlobsC)
                        if GIMetaData['repetition'] == False and GIMetaData['oneToOne'] == True:
                            #print("Checking Migration")
                            score, GHScore, GIScore = Tx.Migration(G,H,choices[i])
                            #print(str(BestTxScore)+","+str(BestTxDetails[0])+","+str(BestTxDetails[1]))
                            #print(str(score)+","+str(GHScore)+","+str(GIScore))
                            if self.AlmostEqual(score,BestTxScore,1):
                                solution = i
                                solSet.append((i,self.GetDeviation(score,BestTxScore)))
                    elif BestTxType == Transformation.RepetitionByExpansion:
                        #print("Checking Rept by Exp")
                        score, xgrowth, ygrowth = Tx.RepetitionByExpansion(H,choices[i])
                        #print("Scores:"+str(score)+":"+str(BestTxScore))
                        if self.AlmostEqual(score,BestTxScore,1):
                            if self.AlmostEqual(xgrowth,BestTxDetails[0],1):
                                if self.AlmostEqual(ygrowth,BestTxDetails[1],1):
                                    solution = i
                                    solSet.append((i,self.GetDeviation(score,BestTxScore)))
                    elif BestTxType == Transformation.RepetitionByTranslation:
                        #print("Checking Rept by Trans")
                        score, leftOffsetCol, leftOffsetRow, rightOffsetCol, rightOffsetRow = Tx.RepetitionByTranslation(H,choices[i])
                        #print("Scores:"+str(score)+":"+str(BestTxScore))
                        if self.AlmostEqual(score,BestTxScore,2):
                            solution = i
                            solSet.append((i,self.GetDeviation(score,BestTxScore)))
                    elif BestTxType == Transformation.BlobTransforms:
                        self.dprint(1,"Checking Blob Transforms")
                        BlobsG = Tx.BlobsA
                        BlobsH = Tx.BlobsB
                        #self.showBlobs(A,BlobsA)
                        BlobsI = Tx.GetBlobs(choices[i])
                        #                   0                           4                   6                   7
                        #details contains (same,morph,translate,scale,addition,deletion,blobCountDiffernce,morph pattern)
                        GHcorresp, GHadditionCount, GHdeletionCount = Tx.GetBlobCorrespondence(BlobsG, BlobsH)
                        HIcorresp, HIadditionCount, HIdeletionCount = Tx.GetBlobCorrespondence(BlobsH, BlobsI)
                        HIBlobMetaData = Tx.GetBlobMetaData(HIcorresp, BlobsH, BlobsI)
                        HInumberMorphed = 0
                        #if HIadditionCount==HIdeletionCount:
                        #    HInumberMorphed = HIadditionCount
                        #    HIadditionCount =0
                        #    HIdeletionCount =0
                        if BestTxDetails[4] == HIadditionCount and BestTxDetails[5] == HIdeletionCount:
                            if HIBlobMetaData['repetition'] == False:
                                if len(HIcorresp.keys()) >= 1:
                                    HIscore, HIsameCount, HImorphCount, HItranslationCount, HIscalingCount = Tx.BlobTransforms(HIcorresp,BlobsH,BlobsI)
                                    HImorphCount += HInumberMorphed

                                    #if morph pattern
                                    if BestTxDetails[7]==1:
                                        GIcorresp, GIadditionCount, GIdeletionCount = Tx.GetBlobCorrespondence(BlobsG, BlobsI)
                                        GInumberMorphed = 0
                                        #if GIadditionCount==GIdeletionCount:
                                        #    GInumberMorphed = HIadditionCount
                                        #    GIadditionCount =0
                                        #    GIdeletionCount =0
                                        GIscore, GIsameCount, GImorphCount, GItranslationCount, GIscalingCount = Tx.BlobTransforms(GIcorresp,BlobsG,BlobsI)
                                        GImorphCount += GInumberMorphed
                                        if HImorphCount == GImorphCount:
                                            if BestTxDetails[1] == HImorphCount and BestTxDetails[2] == HItranslationCount and BestTxDetails[3] == HIscalingCount:
                                                #BestTxDetails[0] == sameCount and
                                                solution = i
                                                solSet.append((i,self.GetDeviation(HIscore,BestTxScore)+self.GetBlobCorrespDeviation(BlobsG,BlobsH,BlobsI,GHcorresp,HIcorresp)))
                                    else:
                                        #if no morph pattern
                                        if BestTxDetails[1] == HImorphCount and BestTxDetails[2] == HItranslationCount and BestTxDetails[3] == HIscalingCount:
                                                #BestTxDetails[0] == sameCount and
                                                solution = i
                                                solSet.append((i,self.GetDeviation(HIscore,BestTxScore)+self.GetBlobCorrespDeviation(BlobsG,BlobsH,BlobsI,GHcorresp,HIcorresp)))

                        #chekcing blob count diff
                        if len(BlobsG)>1 and len(BlobsH)>1:
                            if BestTxDetails[6] == HIBlobMetaData['blobCountDifference']:
                                gAvgFill = self.GetFigAvgFill(BlobsG)
                                hAvgFill = self.GetFigAvgFill(BlobsH)
                                iAvgFill = self.GetFigAvgFill(BlobsI)
                                if not self.AlmostEqual(iAvgFill,gAvgFill,0.1) and not self.AlmostEqual(iAvgFill,hAvgFill,0.1):
                                    solution = i
                                    #solSet.append((i,100-(50*abs(gAvgFill-iAvgFill)+50*abs(hAvgFill-iAvgFill))))
                                    solSet.append((i,0))
                    elif BestTxType == Transformation.ScalingOfOneObject:
                        #print("Chekcing scaling of one obj")
                        BlobsH = Tx.BlobsB
                        BlobsI = Tx.GetBlobs(choices[i])
                        corresp, HIadditionCount, HIdeletionCount = Tx.GetBlobCorrespondence(BlobsH, BlobsI)
                        BlobMetaData = Tx.GetBlobMetaData(HIcorresp, BlobsH, BlobsI)
                        score, widthScale, heightScale = Tx.ScalingOfOneObject(corresp,BlobsH,BlobsI)
                        #print("Scores:"+str(score)+":"+str(BestTxScore))
                        #print("Width scale:"+str(widthScale)+":"+str(BestTxDetails[0]))
                        #print("Height scale:"+str(heightScale)+":"+str(BestTxDetails[1]))
                        if self.AlmostEqual(score,BestTxScore,1):
                            if self.AlmostEqual(widthScale,BestTxDetails[0],0.5):
                                if self.AlmostEqual(heightScale,BestTxDetails[1],0.5):
                                    diff = 0
                                    for t in BlobMetaData['fillComparison'][:]:
                                        #print("t:"+str(t[2]))
                                        diff = diff + t[2]
                                    #print("Diff:"+str(diff))
                                    if self.AlmostEqual(diff,0,3):
                                        solution = i
                                        solSet.append((i,self.GetDeviation(score,BestTxScore)))
                    elif BestTxType == Transformation.TranslationOfOneObject:
                        BlobsH = Tx.BlobsB
                        BlobsI = Tx.GetBlobs(choices[i])
                        corresp, HIadditionCount, HIdeletionCount = Tx.GetBlobCorrespondence(BlobsH, BlobsI)
                        #BlobMetaData = Tx.GetBlobMetaData(HIcorresp, BlobsH, BlobsI)
                        #print("Checking translation of 1 obj")
                        score, data = Tx.TranslationOfOneObject(corresp,BlobsH,BlobsI)
                        #print("Scores:"+str(score)+":"+str(BestTxScore))
                        if self.AlmostEqual(score,BestTxScore,2):
                            listOffsetOrg=[]
                            listOffsetNew = []
                            #print("org")
                            for t in BestTxDetails[:]:
                                listOffsetOrg.append(t[0][2])
                                listOffsetOrg.append(t[0][3])
                                #print(":"+str(t[0][2])+","+str(t[0][3]))
                            #print("new")
                            for t in data[:]:
                                listOffsetNew.append(t[2])
                                listOffsetNew.append(t[3])
                                #print(":"+str(t[2])+","+str(t[3]))
                            listOffsetOrg.sort()
                            listOffsetNew.sort()
                            #print(listOffsetOrg)
                            #print(listOffsetNew)
                            diff = 0
                            for k in range(len(listOffsetOrg)):
                                #print(str(listOffsetNew[k])+","+str(listOffsetOrg[k]))
                                if self.AlmostEqual(listOffsetNew[k],listOffsetOrg[k],2):
                                    pass
                                else:
                                    #print("added diff")
                                    diff = diff + 1
                            if diff == 0:
                                solution = i
                                solSet.append((i,self.GetDeviation(score,BestTxScore)))
        return solSet

    def AnalyseSolutionSet(self, HorSolSet, VerSolSet):
        if len(VerSolSet) == 0:
            if len(HorSolSet) > 0:
                return self.GetBestSolution(HorSolSet)
            else:
                return -1
        else:
            return self.GetBestSolution(VerSolSet)

    def GetBestSolution(self, solSet):
        if len(solSet) > 0:
            solution = solSet[0][0]
            minDeviation = solSet[0][1]
            for t in solSet[:]:
                if t[1]<=minDeviation:
                    solution = t[0]
                    minDeviation = t[1]
            return solution
        else:
            return -1

    def GetDeviation(self,val1,val2):
        return abs(val1-val2)

    def GetBlobCorrespDeviation(self,G,H,I,GHcorresp,HIcorresp):
        GHfillDeviation = 0
        HIfillDeviation = 0
        for k,v in GHcorresp.items():
            GHfillDeviation += abs(G[k].fill - H[v[0][0]].fill)
        for k,v in HIcorresp.items():
            HIfillDeviation += abs(H[k].fill - I[v[0][0]].fill)
        return abs(HIfillDeviation-GHfillDeviation)
        pass

    def GetFigAvgFill(self,BlobG):
        totfill = 0
        for bg in BlobG[:]:
            totfill += bg.fill
        return totfill/len(BlobG)

    def AlmostEqual(self,val1,val2,deviation=0):
        if val1>= val2-deviation and val1<=val2+deviation:
            return True
        else:
            return False

    def FindWholeTransforms(self,A,B,C,D,E,F,G,H,answerChoices):
        Tf = TransformationFinder()
        choices = []
        choices.append(0)
        for option in answerChoices:
            choices.append(self.ToBinary(option))
        solution = -1
        if solution == -1:
            axb = ImageChops.logical_xor(A,B)
            axbxc = ImageChops.logical_xor(axb,C)
            dxe = ImageChops.logical_xor(D,E)
            dxexf = ImageChops.logical_xor(dxe,F)
            score = Tf.Similarity(axbxc,dxexf)
            if score > 96:
                gxh = ImageChops.logical_xor(G,H)
                for i in range(1,len(choices)):
                    gxhxi = ImageChops.logical_xor(gxh,choices[i])
                    s = Tf.Similarity(axbxc,gxhxi)
                    if s > 95:
                        return i
        return solution

    def ToBinary(self, a):
        image_file = Image.open(a)  # open colour image
        image_file = image_file.convert('1')  # convert image to black and white
        image_file = ImageChops.invert(image_file)
        #new_name = "A.png"
        #image_file.save(new_name)
        return image_file

    def dprint(self, level, sent):
        if self.DEBUG:
            if level <= self.debugLevel:
                print(sent)

    def dispTx(self,txs):
        print("SuperTx:")
        tx = txs[0]
        print("=>"+str(tx.txType))
        for k,v in tx.txScores.items():
            print(str(k)+":"+str(v))
        print(str(tx.txDetails))
        print("ImageTx")
        for tx in txs[1]:
            print("=>"+str(tx.txType))
            for k,v in tx.txScores.items():
                print(str(k)+":"+str(v))
            print(str(tx.txDetails))
            print(".")
        print("BlobTx")
        for tx in txs[2]:
            print("=>"+str(tx.txType))
            for k,v in tx.txScores.items():
                print(str(k)+":"+str(v))
            print(str(tx.txDetails))
            print(".")
        return


class DefinitionProblem:
    def __init__(self):
        pass

    @staticmethod
    def is_twobytwo_problem(problem):
        return problem.problemType == '2x2'

    @staticmethod
    def is_threebythree_problem(problem):
        return problem.problemType == '3x3'

    @staticmethod
    def get_problem_figures(problem):
        problem_figures = []
        possible_files = ['A.png', 'B.png', 'C.png', 'D.png', 'E.png', 'F.png', 'G.png', 'H.png']
        for figure_name in problem.figures:
            file_name = problem.figures[figure_name].visualFilename
            if file_name[-5:] in possible_files:
                problem_figures.append(problem.figures[figure_name])
        return problem_figures

    @staticmethod
    def get_solution_figures(problem):
        solution_figures = []
        possible_files = ['1.png', '2.png', '3.png', '4.png', '5.png', '6.png']
        for figure_name in problem.figures:
            file_name = problem.figures[figure_name].visualFilename
            if file_name[-5:] in possible_files:
                solution_figures.append(problem.figures[figure_name])
        return solution_figures

class ImageNameFinder:

    def __init__(self):
        self.image_processor = ImageProcessor()

    # Use root_mean_square_diff < 965 as a similarity condition for search
    def find_matching_image_filename(self, guess_image, solution_images):
        for figure in solution_images:
            file_name = figure.visualFilename
            figure_image = Image.open(file_name)
            if self.image_processor.measure_diff(guess_image, figure_image) < 965:
                return file_name


class ImageProcessor:

    def __init__(self):
        pass

    @staticmethod
    def same_images(image1, image2):
        return ImageChops.difference(image1, image2).getbbox() is None

    @staticmethod
    def measure_diff(image1, image2):
        histogram_diff = ImageChops.difference(image1, image2).histogram()
        sum_of_squares = sum(value * (idx ** 2) for idx, value in enumerate(histogram_diff))
        root_mean_square_diff = (sum_of_squares / float(image1.size[0] * image2.size[1])) ** 0.5
        return root_mean_square_diff

    @staticmethod
    def measure_black_white_ratio(image):
        pixels = image.getdata()
        n_white = 0
        for pixel in pixels:
            if pixel == (255, 255, 255, 255):
                n_white += 1
        n = len(pixels)
        return n_white / float(n)

    @staticmethod
    def black_pixel_distance(image, direction, metric):
        if direction == "right":
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        elif direction == "top":
            image = image.rotate(90)
        elif direction == "bottom":
            image = image.rotate(-90)
        pixels = image.load()
        distances = []
        for i in range(0, image.size[0]):
            for j in range(0, image.size[1]):
                pixel = pixels[i, j]
                # If a non-white pixel is reached, return it
                if pixel != (255, 255, 255, 255):
                    distances.append(j)
                    break
        if len(distances) != 0:
            if metric == "min":
                return min(distances)
            elif metric == "max":
                return max(distances)
            elif metric == "avg":
                return sum(distances) / len(distances)
        else:
            return 0


class Images:

    def __init__(self, image):
        self.image = image
        self.image_processor = ImageProcessor()

    # specify the black_pixel_distance with max and min value for left, right, top, and bottom directions
    def generate_image_analysis(self):
        self.black_white_ratio = self.image_processor.measure_black_white_ratio(self.image)
        self.left_min = self.image_processor.black_pixel_distance(self.image, "left", "min")
        self.right_min = self.image_processor.black_pixel_distance(self.image, "right", "min")
        self.top_min = self.image_processor.black_pixel_distance(self.image, "top", "min")
        self.bottom_min = self.image_processor.black_pixel_distance(self.image, "bottom", "min")

        self.left_max = self.image_processor.black_pixel_distance(self.image, "left", "max")
        self.right_max = self.image_processor.black_pixel_distance(self.image, "right", "max")
        self.top_max = self.image_processor.black_pixel_distance(self.image, "top", "max")
        self.bottom_max = self.image_processor.black_pixel_distance(self.image, "bottom", "max")

        self.left_avg = self.image_processor.black_pixel_distance(self.image, "left", "avg")
        self.right_avg = self.image_processor.black_pixel_distance(self.image, "right", "avg")
        self.top_avg = self.image_processor.black_pixel_distance(self.image, "top", "avg")
        self.bottom_avg = self.image_processor.black_pixel_distance(self.image, "bottom", "avg")

    # set attributes for the key-value set
    def get_attributes(self):
        attributes = {'black_white_ratio': self.black_white_ratio, 'left_min': self.left_min,
                      'right_min': self.right_min, 'top_min': self.top_min, 'bottom_min': self.bottom_min,
                      'left_max': self.left_max, 'right_max': self.right_max, 'top_max': self.top_max,
                      'bottom_max': self.bottom_max, 'left_avg': self.left_avg, 'right_avg': self.right_avg,
                      'top_avg': self.top_avg, 'bottom_avg': self.bottom_avg}

        return attributes

    # Return percentage difference between images for each attribute
    def image_transformation(self, other_image):
        transformation = {}
        current_image_attrs = self.get_attributes()
        for key, value in other_image.get_attributes().items():
            # If pixel distance is within 2 pixels, consider it 100% the same
            if type(value) == int and current_image_attrs[key] - value < 3:
                transformation[key] = 1.0
            else:
                transformation[key] = float(current_image_attrs[key]) / float(value)

        return transformation

    def predict_transformed_attrs(self, transformation_analysis, optional_transformation_analysis=None):
        transformed_attrs = {}
        optional_transformed_attrs = {}
        for key, attr in self.get_attributes().items():
            if optional_transformed_attrs:
                new_value = ((attr * transformation_analysis[key]) + (attr * optional_transformation_analysis[key])) / 2
            else:
                new_value = attr * transformation_analysis[key]
            transformed_attrs[key] = new_value

        return transformed_attrs


class DeeperTransformationAnalysis:

    def __init__(self):
        pass

    @staticmethod
    def generate_similarity_score(solution_attrs, d_expected_attributes):
        scores = []
        for s_key, s_attr in solution_attrs.items():
            # Ignore cases where values are 0
            if s_attr == 0 or d_expected_attributes[s_key] == 0:
                continue
            attr_score = s_attr / d_expected_attributes[s_key]
            scores.append(attr_score)
        score = sum(scores) / len(scores)

        return score

    @staticmethod
    def analyze_transformation_patterns(problem_cells, attributes):
        transformation_patterns = {}
        abc_pattern = {}
        ab_pattern = {}
        ac_pattern = {}
        for attr in attributes:
            a_value = problem_cells['a'].get_attributes()[attr]
            b_value = problem_cells['b'].get_attributes()[attr]
            c_value = problem_cells['c'].get_attributes()[attr]

            if a_value == b_value == c_value:
                abc_pattern[attr] = a_value
            else:
                if a_value == b_value:
                    ab_pattern[attr] = c_value
                elif a_value == c_value:
                    ac_pattern[attr] = b_value

            # For close enough pixel distances, will also have a check
            if type(a_value) == int:
                if (a_value == b_value + 1 == c_value + 1 or a_value == b_value - 1 == c_value - 1
                        or a_value == b_value + 1 == c_value - 1 or a_value == b_value - 1 == c_value + 1
                        or a_value == b_value + 2 == c_value + 2 or a_value == b_value - 2 == c_value - 2
                        or a_value == b_value + 2 == c_value - 2 or a_value == b_value - 2 == c_value + 2):
                    abc_pattern[attr] = a_value
                else:
                    if (a_value == b_value or a_value == b_value + 1 or a_value == b_value - 1
                            or a_value == b_value + 2 or a_value == b_value - 2):
                        ab_pattern[attr] = c_value  # Expect C value in cell D
                    elif (a_value == c_value or a_value == c_value + 1 or a_value == c_value - 1
                          or a_value == c_value + 2 or a_value == c_value - 2):
                        ac_pattern[attr] = b_value  # Expect B value in cell D
            else:  # for close enough black/white ratios
                if round(a_value, 5) == round(b_value, 5) == round(c_value, 5):
                    abc_pattern[attr] = a_value
                else:
                    if round(a_value, 5) == round(b_value, 5):
                        ab_pattern[attr] = c_value  # Expect C value in cell D
                    elif round(a_value, 5) == round(c_value, 5):
                        ac_pattern[attr] = b_value  # Expect B value in cell D
        transformation_patterns['abc'] = abc_pattern
        transformation_patterns['ab'] = ab_pattern
        transformation_patterns['ac'] = ac_pattern

        return transformation_patterns


class TwoByTwoSolver:

    def __init__(self, problem):
        self.image_analyzer = ImageProcessor()
        self.problem = problem

    def check_if_all_equal(self, problem_images):
        cell_images = []
        for image in problem_images:
            figure_image = Image.open(image.visualFilename)
            cell_images.append(figure_image)
        for i in range(len(cell_images) - 1):
            if not self.image_analyzer.measure_diff(cell_images[i], cell_images[i + 1]) < 965:
                return False
        return True

    def check_if_ab_equal(self, problem_images):
        cell_images = {}
        for figure in problem_images:
            figure_image = Image.open(figure.visualFilename)
            cell_images[figure.visualFilename[-5]] = figure_image
        if self.image_analyzer.measure_diff(cell_images['A'], cell_images['B']) < 965:
            return True
        else:
            return False

    def check_if_ac_equal(self, problem_images):
        cell_images = {}
        for figure in problem_images:
            figure_image = Image.open(figure.visualFilename)
            cell_images[figure.visualFilename[-5]] = figure_image
        if self.image_analyzer.measure_diff(cell_images['A'], cell_images['C']) < 965:
            return True
        else:
            return False

    # Checks if image is flipped on it's y-axis when crossing the y-axis of the matrix
    def check_if_y_axis_flip(self, problem_images):
        cell_images = {}
        for figure in problem_images:
            file_name = figure.visualFilename
            figure_image = Image.open(file_name)
            cell_images[file_name[-5]] = figure_image
        # Check if flipping figure A across the y-axis produces figure B
        if self.image_analyzer.measure_diff(cell_images['A'].transpose(Image.FLIP_LEFT_RIGHT),
                                            cell_images['B']) < 965:
            return True
        else:
            return False

    # Checks if image is flipped on it's x-axis when crossing the x-axis of the matrix
    def check_if_x_axis_flip(self, problem_images):
        cell_images = {}
        for figure in problem_images:
            file_name = figure.visualFilename
            figure_image = Image.open(file_name)
            cell_images[file_name[-5]] = figure_image
        # Check if flipping figure A across the y-axis produces figure C
        if self.image_analyzer.measure_diff(cell_images['A'].transpose(Image.FLIP_TOP_BOTTOM),
                                            cell_images['C']) < 965:
            return True
        else:
            return False

    # for basic 2x2 patterns, check the if solution figure match the 5 patterns
    def do_basic_analysis(self):
        problems = DefinitionProblem()
        problem_figures = problems.get_problem_figures(self.problem)
        solution_figures = problems.get_solution_figures(self.problem)
        figure_finder = ImageNameFinder()

        # All equal
        if self.check_if_all_equal(problem_figures):
            example_figure_image = Image.open(problem_figures[0].visualFilename)  # Pass in any figure
            solution_figure_filename = figure_finder.find_matching_image_filename(example_figure_image,
                                                                                  solution_figures)
            if solution_figure_filename:
                solution_number = solution_figure_filename[-5]  # Get image number from filename string
                return int(solution_number)

        # AB equal
        if self.check_if_ab_equal(problem_figures):
            for f in problem_figures:
                if "C.png" in f.visualFilename:
                    f_image = Image.open(f.visualFilename)
                    solution_figure_filename = figure_finder.find_matching_image_filename(f_image,
                                                                                          solution_figures)  # Pass in figure C

                    # If proposed solution figure is present, return the number of that image
                    if solution_figure_filename:
                        solution_number = int(solution_figure_filename[-5])  # Get image number from filename string
                        return solution_number

        # AC equal
        if self.check_if_ac_equal(problem_figures):
            for f in problem_figures:
                if "B.png" in f.visualFilename:
                    f_image = Image.open(f.visualFilename)
                    solution_figure_filename = figure_finder.find_matching_image_filename(f_image,
                                                                                          solution_figures)  # Pass in figure C

                    # If proposed solution figure is present, return the number of that image
                    if solution_figure_filename:
                        solution_number = int(solution_figure_filename[-5])  # Get image number from filename string
                        return solution_number

        # x-axis flip transformation
        if self.check_if_x_axis_flip(problem_figures):
            for f in problem_figures:
                if "B.png" in f.visualFilename:
                    f_image = Image.open(f.visualFilename).transpose(Image.FLIP_TOP_BOTTOM)
                    solution_figure_filename = figure_finder.find_matching_image_filename(f_image,
                                                                                          solution_figures)  # Pass in figure B

                    # If proposed solution figure is present, return the number of that image
                    if solution_figure_filename:
                        solution_number = int(solution_figure_filename[-5])  # Get image number from filename string
                        return solution_number
        # y-axis flip transformation
        if self.check_if_y_axis_flip(problem_figures):
            for f in problem_figures:
                if "C.png" in f.visualFilename:
                    f_image = Image.open(f.visualFilename).transpose(Image.FLIP_LEFT_RIGHT)
                    solution_figure_filename = figure_finder.find_matching_image_filename(f_image,
                                                                                          solution_figures)  # Pass in figure C

                    # If proposed solution figure is present, return the number of that image
                    if solution_figure_filename:
                        solution_number = int(solution_figure_filename[-5])  # Get image number from filename string
                        return solution_number

        return -1

    # Keep solution figures that do match the above observed patterns
    def Keep_valid_solution_figures(self, solution_figures, patterns):
        sifted_solution_figures = solution_figures
        for p_key, pattern in patterns.items():
            for s_key, solution in solution_figures.items():
                solution_attrs = solution.get_attributes()
                valid = False
                # If solution cell attribute is equal to the pattern, add it to sifted_solution_figures
                if round(solution_attrs[p_key], 5) == round(pattern, 5):
                    valid = True
                else:
                    # Otherwise, if solution is a very close int, consider it acceptable
                    if type(solution_attrs[p_key]) == int:
                        if solution_attrs[p_key] == pattern + 1 or solution_attrs[p_key] == pattern + 2:
                            valid = True
                        if solution_attrs[p_key] == pattern - 1 or solution_attrs[p_key] == pattern - 2:
                            valid = True
                    # Otherwise, solution is missing a pattern, so remove it
                    if valid == False:
                        sifted_solution_figures.pop(s_key, None)

        return sifted_solution_figures

    # A detailed transformation analysis on multiple images.
    # Returns solution number if certainty is above the appropriate, else returns -1.
    def do_transformation_analysis(self):
        problems = DefinitionProblem()
        problem_figures = problems.get_problem_figures(self.problem)
        solution_figures = problems.get_solution_figures(self.problem)

        ATTRIBUTES = ['black_white_ratio', 'left_min', 'right_min', 'top_min', 'bottom_min',
                      'left_max', 'right_max', 'top_max', 'bottom_max',
                      'left_avg', 'right_avg', 'top_avg', 'bottom_avg']

        image_problems = {}
        for figure in problem_figures:
            if "A.png" in figure.visualFilename:
                image_problems['a'] = Images(Image.open(figure.visualFilename))
            elif "B.png" in figure.visualFilename:
                image_problems['b'] = Images(Image.open(figure.visualFilename))
            elif "C.png" in figure.visualFilename:
                image_problems['c'] = Images(Image.open(figure.visualFilename))

        image_solutions = {}
        for figure in solution_figures:
            if "1.png" in figure.visualFilename:
                image_solutions['1'] = Images(Image.open(figure.visualFilename))
            elif "2.png" in figure.visualFilename:
                image_solutions['2'] = Images(Image.open(figure.visualFilename))
            elif "3.png" in figure.visualFilename:
                image_solutions['3'] = Images(Image.open(figure.visualFilename))
            elif "4.png" in figure.visualFilename:
                image_solutions['4'] = Images(Image.open(figure.visualFilename))
            elif "5.png" in figure.visualFilename:
                image_solutions['5'] = Images(Image.open(figure.visualFilename))
            elif "6.png" in figure.visualFilename:
                image_solutions['6'] = Images(Image.open(figure.visualFilename))

        for _, problem_cell in image_problems.items():
            problem_cell.generate_image_analysis()

        for _, solution_cell in image_solutions.items():
            solution_cell.generate_image_analysis()

        # Find ABC, AB, AC transformations where attributes remain close to the same
        deeper_analysis = DeeperTransformationAnalysis()
        transformation_patterns = deeper_analysis.analyze_transformation_patterns(image_problems, ATTRIBUTES)
        ABC_pattern = transformation_patterns['abc']
        AB_pattern = transformation_patterns['ab']
        AC_pattern = transformation_patterns['ac']

        # If any ABC attribute or AB, AC transormation attribute remains exactly the same,
        # that attributes value should be expected in the solution.
        # Remove possible solutions cells that do not match these patterns.
        unsifted_images = dict(image_solutions)
        sifted_solution_images = self.Keep_valid_solution_figures(unsifted_images, ABC_pattern)  # Sift for ABC pattern
        sifted_solution_images = self.Keep_valid_solution_figures(sifted_solution_images, AB_pattern)  # Sift for AB pattern
        sifted_solution_images = self.Keep_valid_solution_figures(sifted_solution_images, AC_pattern)  # Sift for AC pattern
        if not sifted_solution_images:
            sifted_solution_images = image_solutions

        # Generate analyses for AB, AC transformations
        ab_trans_analysis = image_problems['a'].image_transformation(image_problems['b'])
        ac_trans_analysis = image_problems['a'].image_transformation(image_problems['b'])

        # Predict the attributes of D that AD, BD, CD transformations will produce based on the observed AB, AC transformations
        abac_to_a = image_problems['a'].predict_transformed_attrs(ab_trans_analysis, ac_trans_analysis)
        ac_to_b = image_problems['b'].predict_transformed_attrs(ac_trans_analysis)
        ab_to_c = image_problems['c'].predict_transformed_attrs(ab_trans_analysis)

        d_expected_attributes = {}
        for attr in ATTRIBUTES:
            total = abac_to_a[attr] + ac_to_b[attr] + ab_to_c[
                attr]
            average = total / 3
            d_expected_attributes[attr] = average

        # Generate similarity scores for each sifted solution
        similarity_scores = {}
        for key, solution in sifted_solution_images.items():
            solution_attrs = solution.get_attributes()
            similarity_score = deeper_analysis.generate_similarity_score(solution_attrs, d_expected_attributes)
            similarity_scores[key] = similarity_score

        # Choose the solution with the highest similarity score
        answer = int(max(similarity_scores, key=similarity_scores.get))
        similarity = similarity_scores[str(answer)]

        # If the combined similarity score is above 95% threshold, then return the number of the  solution image,
        # otherwise, if sift out enough solutions, return the solution with the highest similarity score,
        # otherwise, return -1
        if answer and similarity > 0.95:
            return answer
        elif len(sifted_solution_images) < 4:
            return answer
        else:
            return -1
