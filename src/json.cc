#include "json.h"
#include "node.h"
#include <assert.h>
#include <rapidjson/document.h>
#include <rapidjson/filestream.h>
#include <rapidjson/writer.h>

#if defined NDEBUG
# define RAPID_JSON_CHECK_HAS_MEMBER(value, member)
#else
# define RAPID_JSON_CHECK_HAS_MEMBER(value, member) \
    do if (!value.HasMember(member)) {fprintf(stderr, "should have member: %s\n", member); return -1;} \
    while (0)
#endif

using namespace rapidjson;

static int load_tree(const Value& tree, TreeNodeBase * node)
{
    if (tree.HasMember("value"))
    {
        node->leaf() = true;
        node->y() = tree["value"].GetDouble();
    }
    else
    {
        node->leaf() = false;

        RAPID_JSON_CHECK_HAS_MEMBER(tree, "split_index");
        RAPID_JSON_CHECK_HAS_MEMBER(tree, "split_type");
        RAPID_JSON_CHECK_HAS_MEMBER(tree, "split_value");
        RAPID_JSON_CHECK_HAS_MEMBER(tree, "left");
        RAPID_JSON_CHECK_HAS_MEMBER(tree, "right");

        node->split_x_index() = tree["split_index"].GetInt();
        const char * type = tree["split_type"].GetString();
        if (strcmp(type, "numerical") == 0)
        {
            node->split_x_type() = kXType_Numerical;
            node->split_x_value().d() = tree["split_value"].GetDouble();
        }
        else if (strcmp(type, "category") == 0)
        {
            node->split_x_type() = kXType_Category;
            node->split_x_value().i() = tree["split_value"].GetInt();
        }
        else
        {
            fprintf(stderr, "invalid type: %s\n", type);
            return -1;
        }

        const Value& left = tree["left"];
        TreeNodeBase * left_node = TreeNodePredictor::create();
        if (load_tree(left, left_node) == -1)
        {
            delete left_node;
            return -1;
        }
        else
        {
            node->left() = left_node;
        }

        const Value& right = tree["right"];
        TreeNodeBase * right_node = TreeNodePredictor::create();
        if (load_tree(right, right_node) == -1)
        {
            delete right_node;
            return -1;
        }
        else
        {
            node->right() = right_node;
        }
    }

    return 0;
}

int load_json(
    FILE * fp,
    double * y0,
    std::vector<TreeNodeBase *> * trees)
{
    assert(trees->empty());
    FileStream stream(fp);
    Document document;
    document.ParseStream<0>(stream);
    if (document.HasParseError())
    {
        fprintf(stderr, "parse json error: %s\n", document.GetParseError());
        return -1;
    }

    RAPID_JSON_CHECK_HAS_MEMBER(document, "y0");
    RAPID_JSON_CHECK_HAS_MEMBER(document, "trees");

    *y0 = document["y0"].GetDouble();

    const Value& _trees = document["trees"];
    for (SizeType i=0, s=_trees.Size(); i<s; i++)
    {
        const Value& tree = _trees[i];
        TreeNodeBase * node = TreeNodePredictor::create();
        if (load_tree(tree, node) == -1)
        {
            delete node;
            for (size_t i=0, s=trees->size(); i<s; i++)
                delete (*trees)[i];
            trees->clear();
            return -1;
        }
        trees->push_back(node);
    }

    return 0;
}

static void save_spec(const XYSpec& spec, Value * spec_value,
                      Document::AllocatorType& allocator)
{
    spec_value->SetArray();
    for (size_t i=0, s=spec.get_x_type_size(); i<s; i++)
    {
        kXType x_type = spec.get_x_type(i);
        spec_value->PushBack((x_type == kXType_Numerical) ? "numerical" : "category",
            allocator);
    }
}

static void save_tree(const TreeNodeBase& tree, Value * tree_value,
                      Document::AllocatorType& allocator)
{
    tree_value->SetObject();
    if (tree.is_leaf())
    {
        tree_value->AddMember("value", tree.y(), allocator);
    }
    else
    {
        tree_value->AddMember("split_index", (int)tree.split_x_index(), allocator);

        bool numerical = tree.split_is_numerical();
        if (numerical)
        {
            tree_value->AddMember("split_type", "numerical", allocator);
            tree_value->AddMember("split_value", tree.split_get_double(), allocator);
        }
        else
        {
            tree_value->AddMember("split_type", "category", allocator);
            tree_value->AddMember("split_value", tree.split_get_int(), allocator);
        }

        Value left_value;
        save_tree(*tree.left(), &left_value, allocator);
        tree_value->AddMember("left", left_value, allocator);

        Value right_value;
        save_tree(*tree.right(), &right_value, allocator);
        tree_value->AddMember("right", right_value, allocator);
    }
}

void save_json(
    FILE * fp,
    const XYSpec& spec,
    double y0,
    const std::vector<TreeNodeBase *>& trees)
{
    FileStream stream(fp);
    Writer<FileStream> writer(stream);

    Document document;
    Document::AllocatorType& allocator = document.GetAllocator();
    document.SetObject();

    Value spec_value;
    save_spec(spec, &spec_value, allocator);
    document.AddMember("spec", spec_value, allocator);

    document.AddMember("y0", y0, allocator);

    Value trees_value;
    trees_value.SetArray();
    for (size_t i=0, s=trees.size(); i<s; i++)
    {
        Value tree_value;
        save_tree(*trees[i], &tree_value, allocator);
        trees_value.PushBack(tree_value, allocator);
    }
    document.AddMember("trees", trees_value, allocator);

    document.Accept(writer);
}
