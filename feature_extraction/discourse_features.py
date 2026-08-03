from isanlp_rst.parser import Parser
from collections import Counter

"""
Following Features are extracted by the Discourse class
- Imbalance of the tree
- Imbalance from the root
- Frequency of each EDU relation
- Frequency of each satellite nucleus combination
- Frequency of each EDU relation + satellite nucleus combination
"""

class DiscourseFeatures():
    def __init__(self, version='rstdt'):
        self.version = version
        self.parser =(
            Parser(hf_model_name='tchewik/isanlp_rst_v3', hf_model_version=version, cuda_device=-1)) # Use -1 for CPU for CUDA
        # make sure we get a dictionary of a fixed size

        self.RST_RELATIONS = ['Contrast_NN', 'Contrast_SN', 'Contrast_NS',
                              'Enablement_NN', 'Enablement_SN', 'Enablement_NS',
                              'Evaluation_NN', 'Evaluation_SN', 'Evaluation_NS',
                              'textual-organization_NN', 'textual-organization_SN', 'textual-organization_NS',
                              'Condition_NN', 'Condition_SN', 'Condition_NS',
                              'Joint_NN', 'Joint_SN', 'Joint_NS',
                              'Background_NN', 'Background_SN', 'Background_NS',
                              'Temporal_NN', 'Temporal_SN', 'Temporal_NS',
                              'Attribution_NN', 'Attribution_SN', 'Attribution_NS',
                              'Manner-Means_NN', 'Manner-Means_SN', 'Manner-Means_NS',
                              'Topic-Comment_NN', 'Topic-Comment_SN', 'Topic-Comment_NS',
                              'Summary_NN', "Summary_SN", "Summary_NS",
                              'Comparison_NN', 'Comparison_SN', 'Comparison_NS',
                              'Elaboration_NN', 'Elaboration_SN', 'Elaboration_NS',
                              'Explanation_NN', 'Explanation_SN', 'Explanation_NS',
                              'Topic-Change_NN', 'Topic-Change_SN', 'Topic-Change_NS',
                              'same-unit_NN', 'same-unit_SN', "same-unit_NS",
                              'Cause_NN', 'Cause_SN', 'Cause_NS']

        self.RST_EDUS = ['Contrast', 'Enablement', 'Evaluation', 'textual-organization', 'Condition', 'Joint',
                         'Background', 'Temporal', 'Attribution', 'Manner-Means', 'Topic-Comment', 'Summary',
                         'Comparison', 'Elaboration', 'Explanation', 'Topic-Change', 'same-unit', 'Cause']
        """
            'Elaboration', 'Attribution', 'Joint', 'same-unit', 'Attribution', 'Explanation', 'Enablement',
                         'Background', 'Evaluation', 'Cause', 'Contrast', 'Contrast', 'Background', 'Temporal',
                         'Comparison', 'Contrast', 'Topic-Change', 'Manner-Means', 'textual-organization', 'Temporal',
                         'Condition', 'Condition', 'Cause', 'Summary', 'Topic-Comment', 'Cause', 'Summary', 'Evaluation',
                         'Temporal', 'Explanation', 'Enablement', 'Topic-Comment', 'Comparison', 'Elaboration',
                         'Manner-Means', 'Comparison', 'Summary', 'Condition', 'Topic-Comment', 'Topic-Change',
                         'Evaluation', 'Explanation'] #
        """
        self.nucleus_satellite = ["NN", "NS", "SN"]

    def run_analysis(self, doc):
        """
        Wrapper of all functions, each functions get passed the parse of the model to calculate the features.
        :param doc:
        :return:
        """
        res = self.parser(doc) # this throws an error if idx is out of bounds
        imbalance_root = self.imbalance_for_root_node(res) # int
        total_imbalance = self.calculate_imbalance(res) # float
        edu_freq, edu_freq_count = self.calculate_edu_relations(res) #dict
        edu_freq_nuc, total_relations_edu_and_satellite = self.calculate_edu_relations_with_nucleus_and_satellite(res) #dict
        nuclearity_freq, total_nuclearity_counts = self.calculate_nuclearity(res) #dict
        return {"imbalance_root": imbalance_root,
                "total_imbalance": total_imbalance,
                "edu_freq": edu_freq,
                "edu_freq_count": edu_freq_count,
                "edu_freq_nuc": edu_freq_nuc,
                "total_relations_edu_and_satellite": total_relations_edu_and_satellite,
                "nuclearity_freq": nuclearity_freq,
                "total_nuclearity_counts": total_nuclearity_counts}

    def imbalance_for_root_node(self, parse) -> int:
        """
        Check recursively, if the tree is balanced
        Check with |L-R| / L+R
        where L is the number ofl eaves under the left child and R under the right child.
        Calculate how many EDUs are contained under left child vs. right child at each branching point
              root
            /    \
        EDU      X
                / \
             EDU  EDU
        L=1, R=2 imb:|1-2|/3 = 0.333
        :param parse:
        :return:
        """
        root = parse['rst'][0]
        # count leaves on the left and right side
        def count_leaves(node):
            # stopping criteria
            if not hasattr(node, "left") or node.left is None:
                return 1
            # recursive case
            return count_leaves(node.left) + count_leaves(node.right)
        left_size = count_leaves(root.left)
        right_size = count_leaves(root.right)
        root_imbalance = abs(left_size-right_size)/(left_size+right_size)
        #print(root_imbalance)
        return root_imbalance

    def calculate_imbalance(self, parse) -> float:
        """
        Check recursively, if the tree is balanced
        Check with |L-R| / L+R
        where L is the number ofl eaves under the left child and R under the right child.
        Calculate how many EDUs are contained under left child vs. right child at each branching point
              root
            /    \
        EDU      X
                / \
             EDU  EDU
        L=1, R=2 imb:|1-2|/3 = 0.333
        :param parse:
        :return:
        see: On the Flatness, Non-linearity, and Branching Direction of Natural Language and Random Constituency Trees:
        Analyzing Structural Variation within and across Languages
        """
        subtree_size = 0
        root = parse['rst'][0]
        def visit(node):
            """
            Calculate the count of leaves at the specific node as well as the previous imbalance scores
            :param node:
            :return:
            """
            # stopping criteria
            if not hasattr(node, "left") or node.left is None:
                return 1, []
            # recursive case
            left_count, left_prev_imbalance_scores = visit(node.left)
            right_count, right_prev_imbalance_scores = visit(node.right)

            # get the current score
            current_score = abs(left_count-right_count)/(left_count+right_count)

            total_count = left_count + right_count
            all_scores = left_prev_imbalance_scores + right_prev_imbalance_scores + [current_score]

            return total_count, all_scores
        _, scores = visit(root)
        return sum(scores)/len(scores) if scores else 0.0

    def calculate_edu_relations(self, res) -> dict:
        """
        Traverse the tree and count the EDU relations.
        :param res:
        :return features: dictioanry
        """
        root = res['rst'][0]
        edu_counter = Counter()
        def visit(node):
            # stopping criteria
            # always fill a binary tree from the left, thats why it is the stopping criteria
            if not hasattr(node, "left") or node.left is None:
                return
            # get the edus from the left and the right side
            edu_counter[node.relation]+=1
            visit(node.left)
            visit(node.right)
        visit(root)
        #print(edu_counter)
        total_relations = sum(edu_counter.values())
        features = {
            relation: edu_counter.get(relation, 0)
            for relation in self.RST_EDUS
        }
        return features, total_relations

    def calculate_edu_relations_with_nucleus_and_satellite(self, res) -> dict:
        """
        Traverse the tree and count the EDU relations.
        :param res:
        :return:
        """
        root = res['rst'][0]
        edu_counter = Counter()
        def visit(node):
            # stopping criteria
            # always fill a binary tree from the left, thats why it is the stopping criteria
            if not hasattr(node, "left") or node.left is None:
                return
            # get the edus from the left and the right side
            relation = node.relation + "_" + node.nuclearity
            edu_counter[relation]+=1
            visit(node.left)
            visit(node.right)
        visit(root)
        #print(edu_counter)
        total_relations = sum(edu_counter.values())
        features = {
            relation: edu_counter.get(relation, 0)
            for relation in self.RST_RELATIONS
        }
        return features, total_relations

    def calculate_nuclearity(self, res) -> dict:
        """
        Traverse the tree and count the EDU relations.
        :param res:
        :return:
        """
        root = res['rst'][0]
        nuclearity_counter = Counter()
        def visit(node):
            # stopping criteria
            # always fill a binary tree from the left, thats why it is the stopping criteria
            if not hasattr(node, "left") or node.left is None:
                return
            # get the edus from the left and the right side
            nuclearity_counter[node.nuclearity]+=1
            visit(node.left)
            visit(node.right)
        visit(root)
        #print(nuclearity_counter)
        total_relations = sum(nuclearity_counter.values())
        features = {
            relation: nuclearity_counter.get(relation, 0)
            for relation in self.nucleus_satellite
        }
        features["NN_"] = features.pop("NN") # rename this value, otherwise it is overwritten with NN from parsing
        return features, total_relations

