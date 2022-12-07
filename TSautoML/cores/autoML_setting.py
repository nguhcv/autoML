from TSautoML.cores.TSdata import TSData
from TSautoML.ML_pipeline.ML_based_pipeline import MLbased_Forecasting_pipeline



class autoML:
    """The object to receive autoML setting from user:
        - deloy autoML pipeline from setting.

    Arguments:
    - dataset: A TS-data object.
     - task: define the tasks
     - approach: NAS or ML
     - metric
    """

    def __init__(
        self,
        dataset: TSData,
        task:str,
        approach:str,
        metric:str,
        saved_path:str,
        train_test_split_ratio:float=0.8,
        tuning_mechanism:str='grid',

    ) -> None:
        self.dataset = dataset
        self.metric = metric
        self.train_test_ratio = train_test_split_ratio
        self.saved_path = saved_path


        if approach=='ML' or approach=='NAS':
            self.approach = approach
            if task=='cl' or task =='ad' or task =='fc':
                self.task = task
                if tuning_mechanism=='grid' or tuning_mechanism =='random':
                    self.tuning = tuning_mechanism
                else:
                    raise Exception('please select: grid or random')
            else:
                raise Exception('please select: cl, ad or fc')
        else:
            raise Exception('please select: ML, NAS')

    def deploy(self):

        if self.approach == 'ML' and self.task =='fc':
            if self.tuning=='grid':
                self.auto_pipeline = MLbased_Forecasting_pipeline(dataset=self.dataset,
                                                        train_test_ratio=self.train_test_ratio,
                                                        save_path=self.saved_path,
                                                        metric=self.metric,
                                                        tuning=self.tuning)
                self.auto_pipeline.run()
            elif self.tuning=='random':
                pass
        else:
            pass

    def best_score(self):
        self.auto_pipeline.best_scores()
        pass

    def plot_result(self):
        self.auto_pipeline.plot_results()
        pass







